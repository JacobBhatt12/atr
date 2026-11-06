// Measures one or more claude.ai chats for the ATR A/B test. Run it in a claude.ai tab
// (DevTools console, or a browser automation tool) after setting CHATS:
//
//   const CHATS = {"A-wiki-1": "<chat uuid>", "C-wiki-1": "<chat uuid>", ...};
//
// It returns one JSON line per chat. Save them all to benchmarks/claude_ai/runs.jsonl and run
// `python -m benchmarks.claude_ai.report`.
//
// claude.ai doesn't expose token counts, so tokens are estimated as characters / 4 (the same
// estimate ATR uses without tiktoken). What it measures:
//   result_tokens   text of every tool result: what the tools put into the chat
//   context_tokens  what the model re-reads over the whole task: at each model call, everything
//                   so far (prompt, earlier replies, tool calls and results). Thinking and the
//                   system prompt are left out (the same for both sides).
//   calls           model calls (one per tool round, plus the final answer)
async function measureChats(CHATS) {
  const org = (await fetch('/api/organizations').then(r => r.json())).find(o => o.capabilities.includes('chat')).uuid;
  const tok = s => Math.ceil((s || '').length / 4);
  const blockText = b => b.type === 'text' ? b.text
    : b.type === 'tool_use' ? JSON.stringify(b.input || {})
    : b.type === 'tool_result' ? (Array.isArray(b.content) ? b.content.map(c => c.text || JSON.stringify(c)).join('\n') : String(b.content || ''))
    : '';
  const out = [];
  for (const [label, uuid] of Object.entries(CHATS)) {
    const j = await fetch(`/api/organizations/${org}/chat_conversations/${uuid}?tree=True&rendering_mode=messages&render_all_tools=true`).then(r => r.json());
    const msgs = j.chat_messages || [];
    const user = msgs.find(m => m.sender === 'human');
    const bot = msgs.filter(m => m.sender === 'assistant').pop();
    const blocks = (bot && bot.content || []).filter(b => b.type !== 'thinking');
    let ctx = tok(user && (user.text || (user.content || []).map(blockText).join(''))), context = 0, calls = 0, results = 0;
    const tools = [];
    for (const b of blocks) {
      if (b.type === 'tool_use') { context += ctx; calls += 1; tools.push((b.integration_name ? b.integration_name + ':' : '') + b.name); }
      const t = tok(blockText(b));
      if (b.type === 'tool_result') results += t;
      ctx += t;
    }
    context += ctx; calls += 1;  // the final answer
    const answer = blocks.filter(b => b.type === 'text').map(b => b.text).join('\n').trim();
    out.push({label, uuid, model: j.model, created_at: j.created_at, tools, calls, result_tokens: results,
              context_tokens: context, answer: answer.slice(-1500)});
  }
  return out.map(o => JSON.stringify(o)).join('\n');
}
