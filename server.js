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

// Helper: Clean and sanitize API keys
function sanitizeKey(key) {
  if (!key) return '';
  return key
    .replace(/^Bearer\s+/i, '') // strip leading "Bearer "
    .replace(/["']/g, '')        // strip quotes
    .trim();
}

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

  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  if (model.includes('/')) {
    return model;
  }

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
    thinking_mode: ENABLE_THINKING_MODE,
    has_env_key: Boolean(NIM_API_KEY)
  });
});

// List models endpoint
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

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  let keepAliveTimer = null;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // 1. Resolve API Key intelligently
    const clientKey = sanitizeKey(req.headers.authorization);
    const envKey = sanitizeKey(NIM_API_KEY);

    let apiKey = '';
    // If the frontend passed a real NVIDIA key starting with "nvapi-", use it
    if (clientKey && clientKey.startsWith('nvapi-')) {
      apiKey = clientKey;
    } else if (envKey) {
      // Otherwise, prioritize the Render environment variable
      apiKey = envKey;
    } else if (clientKey) {
      apiKey = clientKey;
    }

    // 2. Validate format before calling NVIDIA
    if (!apiKey || !apiKey.startsWith('nvapi-')) {
      const hint = apiKey ? `Received value starting with: "${apiKey.substring(0, 10)}..."` : 'No key provided';
      console.error(`Auth Error: NVIDIA keys must start with "nvapi-". ${hint}`);
      return res.status(401).json({
        error: {
          message: `Invalid NVIDIA API key. Keys must start with 'nvapi-'. ${hint}. Check your NIM_API_KEY in Render or your frontend API key setting.`,
          type: 'authentication_error',
          code: 401
        }
      });
    }

    const nimModel = resolveNimModel(model);

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

    // Call NVIDIA NIM
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Accept': stream ? 'text/event-stre
