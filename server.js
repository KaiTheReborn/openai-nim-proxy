// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = process.env.SHOW_REASONING === 'true' || false;

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true' || false;

// Model mapping
const MODEL_MAPPING = {
  'bds4': 'deepseek-ai/deepseek-v4-pro-0813',
  'misty': 'mistralai/mistral-large-4-1t-a52b',
  'dsv4': 'deepseek-ai/deepseek-v4.1-flash',
  'glms5': 'z-ai/glm-5.3-flash',
  'qwen': 'qwen/Qwen3.8-2.4T-A95B',
  'kim': 'moonshotai/kimi-k3',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking' 
};

// Helper: Resolve model name without blocking network calls
function resolveNimModel(model) {
  if (!model) return 'meta/llama-3.1-8b-instruct';

  // 1. Check direct alias in mapping
  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  // 2. If it's already a full NVIDIA NIM model ID (e.g., "meta/llama-3.1-70b-instruct"), pass it through
  if (model.includes('/')) {
    return model;
  }

  // 3. Fallbacks based on common OpenAI/Claude model names
  const lower = model.toLowerCase();
  if (lower.includes('405b') || lower.includes('opus')) {
    return 'meta/llama-3.1-405b-instruct';
  } else if (lower.includes('70b') || lower.includes('gpt-4') || lower.includes('claude')) {
    return 'meta/llama-3.1-70b-instruct';
  }

  return 'meta/llama-3.1-8b-instruct';
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  let keepAliveTimer = null;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Resolve API key (supports Render env variable OR Bearer token from client)
    const clientAuth = req.headers.authorization;
    const apiKey = (clientAuth && clientAuth.startsWith('Bearer '))
      ? clientAuth.replace('Bearer ', '').trim()
      : NIM_API_KEY;

    if (!apiKey) {
      return res.status(401).json({
        error: {
          message: 'No NVIDIA API key provided. Set NIM_API_KEY in Render or pass Bearer token.',
          type: 'invalid_request_error',
          code: 401
        }
      });
    }

    const nimModel = resolveNimModel(model);

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: max_tokens || 4096,
      stream: Boolean(stream)
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    // Axios request with a 5-minute timeout to avoid premature drops
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000 // 5 minutes
    });

    if (stream) {
      // Set SSE headers and flush immediately so Render/proxies don't timeout
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      if (res.flushHeaders) res.flushHeaders();

      // Send periodic keep-alive comments while waiting for the model
      keepAliveTimer = setInterval(() => {
        if (!res.writableEnded) {
          res.write(': keep-alive\n\n');
        }
      }, 15000);

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          const trimmed = line.trim();
          if (!trimmed) return;

          if (trimmed.startsWith('data: ')) {
            if (trimmed.includes('[DONE]')) {
              res.write('data: [DONE]\n\n');
              return;
            }

            try {
              const data = JSON.parse(trimmed.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                const reasoning = delta.reasoning_content;
                const content = delta.content;

                if (SHOW_REASONING) {
                  let combined = '';
                  if (reasoning && !reasoningStarted) {
                    combined = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combined = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combined += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combined += content;
                  }

                  if (combined) {
                    delta.content = combined;
                  }
                } else {
                  if (!content) {
                    delta.content = '';
                  }
                }
                delete delta.reasoning_content;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(`${trimmed}\n\n`);
            }
          }
        });
      });

      response.data.on('end', () => {
        if (keepAliveTimer) clearInterval(keepAliveTimer);
        if (reasoningStarted) {
          res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n</think>\n\n' } }] })}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      response.data.on('error', (err) => {
        if (keepAliveTimer) clearInterval(keepAliveTimer);
        console.error('NIM stream error:', err.message);
        if (!res.writableEnded) res.end();
      });

      req.on('close', () => {
        if (keepAliveTimer) clearInterval(keepAliveTimer);
        if (response.data?.destroy) response.data.destroy();
      });

    } else {
      // Non-streaming response
      const choice = response.data.choices?.[0] || {};
      let fullContent = choice.message?.content || '';

      if (SHOW_REASONING && choice.message?.reasoning_content) {
        fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
      }

      res.json({
        id: response.data.id || `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: response.data.created || Math.floor(Date.now() / 1000),
        model: model,
        choices: [
          {
            index: 0,
            message: {
              role: choice.message?.role || 'assistant',
              content: fullContent
            },
            finish_reason: choice.finish_reason || 'stop'
          }
        ],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (error) {
    if (keepAliveTimer) clearInterval(keepAliveTimer);

    const status = error.response?.status || 500;
    const errorDetails = error.response?.data || error.message;
    console.error(`Proxy error [${status}]:`, errorDetails);

    if (!res.headersSent) {
      res.status(status).json({
        error: {
          message: typeof errorDetails === 'object' ? JSON.stringify(errorDetails) : errorDetails,
          type: 'proxy_error',
          code: status
        }
      });
    } else {
      res.end();
    }
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, code: 404 } });
});

// Configure Node HTTP Server timeouts
const server = app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});

server.timeout = 300000;         // 5 minutes
server.keepAliveTimeout = 120000; // 2 minutes
server.headersTimeout = 125000;
