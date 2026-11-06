"""The hosted connector landing page and personal-link builder."""
from __future__ import annotations

import html

# Brand SVG paths sourced from https://github.com/pheralb/svgl/tree/main/static/library
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#080a08">
<title>ATR | Less Tokens, More Usage</title>
<meta name="description" content="ATR is a free add-on for Claude, ChatGPT and other AI apps. When your AI reads a web page, a document or data from apps like GitHub, ATR gives it only the part that answers your question.">
<style>
:root { color-scheme:dark; --bg:oklch(11% .007 145); --surface:oklch(15% .009 145); --ink:oklch(96% .009 145); --muted:oklch(69% .015 145); --line:oklch(27% .023 145); --soft:oklch(18% .017 145); --green:oklch(88% .25 140); }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:32px; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; -webkit-font-smoothing:antialiased; }
a { color:inherit; text-underline-offset:4px; }
button,input { font:inherit; }
button,a,input,summary { -webkit-tap-highlight-color:transparent; }
:focus-visible { outline:2px solid var(--green); outline-offset:5px; }
.wrap { width:min(1160px,calc(100% - 80px)); margin-inline:auto; }
.announcement { text-align:center; padding:9px 20px; font-size:12px; }
.announcement a { text-decoration:none; }
.announcement span { margin-left:10px; }
nav { display:flex; align-items:center; justify-content:space-between; min-height:96px; gap:24px; }
.brand { display:inline-flex; align-items:center; gap:9px; font-size:25px; letter-spacing:-1px; font-weight:650; text-decoration:none; }
.navlinks { display:flex; align-items:center; gap:28px; font-size:13px; }
.navlinks a { text-decoration:none; }
.button,button { display:inline-flex; align-items:center; justify-content:center; gap:24px; padding:12px 20px; border:1px solid var(--ink); background:var(--ink); color:var(--surface); text-decoration:none; font-size:13px; cursor:pointer; border-radius:2px; transition:background .2s,transform .2s; }
.button:hover,button:hover { background:var(--green); }.button:active,button:active { transform:translateY(1px); }
.button.secondary,button.ghost { background:transparent; color:var(--ink); border-color:var(--line); }.button.secondary:hover,button.ghost:hover { background:var(--soft); }
.hero { text-align:center; padding:76px 0 64px; }
.eyebrow { display:inline-flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); margin:0 0 24px; }
.dot { width:6px; height:6px; background:var(--green); border-radius:50%; }
h1,h2,h3,p { margin-top:0; }
h1,h2 { font-family:Georgia,"Times New Roman",serif; font-weight:400; }
h1 { font-size:clamp(48px,6.3vw,84px); letter-spacing:-.06em; line-height:1.08; margin:0 0 26px; }
h1 span { color:var(--muted); }
.hero .lede { max-width:540px; margin:0 auto 28px; color:var(--muted); font-size:16px; line-height:1.75; }
.hero .button { min-width:170px; }.hero .fine { margin:28px 0 0; font-size:11px; color:var(--muted); }
.preview { border:1px solid var(--line); background:var(--surface); box-shadow:0 22px 60px -40px oklch(25% .01 100 / .35); }
.preview-top { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:17px 24px; border-bottom:1px solid var(--line); font-size:12px; }
.preview-top strong { font-weight:500; }.preview-top span { color:var(--muted); }
.preview-body { display:grid; grid-template-columns:180px 1fr; min-height:370px; }
.sidebar { border-right:1px solid var(--line); padding:25px 15px; font-size:12px; color:var(--muted); }
.sidebar p { margin:0 0 5px; padding:9px 13px; }.sidebar .selected { color:var(--ink); background:var(--soft); }.sidebar .small { margin-top:66px; font-size:10px; }
.dashboard { padding:30px 35px; }.dash-heading { display:flex; align-items:center; justify-content:space-between; gap:15px; margin-bottom:30px; }.dash-heading h3 { font-size:17px; font-weight:500; margin:0; }.tag { font-size:10px; border:1px solid var(--line); padding:4px 8px; color:var(--muted); }
.chart { display:grid; grid-template-columns:1fr 1fr; gap:36px; }.chart p { font-size:11px; color:var(--muted); margin-bottom:4px; }.chart strong { font-size:32px; font-weight:450; letter-spacing:-1px; }.chart strong small { font-size:11px; letter-spacing:0; font-weight:400; color:var(--muted); }
.bars { display:flex; align-items:flex-end; height:90px; gap:5px; margin-top:20px; border-bottom:1px solid var(--line); }
.bars i { flex:1; height:var(--h); background:repeating-linear-gradient(135deg,var(--line) 0,var(--line) 1px,transparent 1px,transparent 4px); border:1px solid var(--line); border-bottom:0; }.bars.reduced i { background:var(--green); border:0; }
.dash-bottom { display:flex; justify-content:space-between; gap:15px; margin-top:24px; font-size:11px; color:var(--muted); }.dash-bottom b { font-weight:500; color:var(--green); }
.caption { text-align:center; font-size:11px; color:var(--muted); margin:16px 0 0; }
.compatibility { text-align:center; padding:64px 0 76px; }.compatibility p { font-size:12px; color:var(--muted); margin-bottom:25px; }.apps { display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:clamp(22px,4vw,58px); font-size:19px; letter-spacing:-.5px; font-weight:500; }.apps > span { display:inline-flex; align-items:center; gap:10px; white-space:nowrap; }.app-logo { display:block; width:28px; height:28px; flex:none; }.apps small { font-size:12px; color:var(--muted); font-weight:400; }
.section { padding:85px 0; border-top:1px solid var(--line); }
h2 { font-size:clamp(34px,4vw,48px); line-height:1.15; letter-spacing:-.045em; margin-bottom:18px; }
.section-intro { max-width:550px; color:var(--muted); margin-bottom:42px; }
.feature { display:grid; grid-template-columns:1fr 1fr; gap:90px; align-items:center; }.feature p { color:var(--muted); max-width:460px; }.feature .text-link { display:inline-block; margin-top:12px; font-size:13px; color:var(--ink); }
.flow { padding:35px; background:var(--soft); border:1px solid var(--line); }.flow-row { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:15px 0; font-size:13px; }.flow-row small { color:var(--muted); font-size:11px; }.flow-arrow { text-align:center; color:var(--muted); line-height:1; }.flow-row.atr { padding:17px 20px; border:1px solid var(--ink); margin:14px 0; background:var(--surface); }
.benefits { display:grid; grid-template-columns:1fr 1fr 1fr; gap:40px; margin-top:60px; }.benefits h3 { font-size:15px; font-weight:500; margin:13px 0 10px; }.benefits p { font-size:13px; color:var(--muted); }.number { font-size:11px; color:var(--muted); }
.setup { display:grid; grid-template-columns:1fr 1.1fr; gap:85px; }.setup .section-intro { margin-bottom:25px; }
.copy { display:flex; gap:8px; align-items:stretch; margin:8px 0; }.url { flex:1; min-width:0; overflow-x:auto; white-space:nowrap; padding:12px; background:var(--surface); border:1px solid var(--line); font-size:12px; }
code,.url { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }code { font-size:.88em; }.muted { color:var(--muted); font-size:12px; }.instructions { border-top:1px solid var(--line); }details { border-bottom:1px solid var(--line); }summary { cursor:pointer; display:flex; justify-content:space-between; align-items:center; padding:18px 0; font-size:14px; list-style:none; }summary::-webkit-details-marker { display:none; }summary::after { content:'+'; font-size:18px; color:var(--muted); }details[open] summary::after { content:'−'; }details p,details ol { font-size:13px; color:var(--muted); margin:0 0 22px; }ol { padding-left:20px; }li { margin-bottom:8px; }
.tools { display:grid; grid-template-columns:1fr 1fr; column-gap:60px; }.tool { padding:24px 0; border-top:1px solid var(--line); }.tool h3 { font-size:14px; font-weight:500; margin-bottom:9px; }.tool p { font-size:13px; color:var(--muted); margin-bottom:0; }
.personal { display:grid; grid-template-columns:.85fr 1.15fr; gap:85px; }.personal p { color:var(--muted); }.card { border:1px solid var(--line); padding:26px; background:var(--surface); }.row { display:grid; grid-template-columns:1fr 2fr; gap:12px; margin-bottom:12px; }label { display:block; font-size:11px; color:var(--muted); margin-bottom:6px; }input { width:100%; min-width:0; padding:10px; border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:1px; font-size:12px; }.server+.server { border-top:1px solid var(--line); margin-top:24px; padding-top:24px; }.server-heading { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; font-size:12px; }.remove { background:transparent; color:var(--muted); border:0; padding:4px; font-size:11px; }.remove:hover { color:var(--ink); background:var(--soft); }.actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:20px; }#result { margin-top:15px; font-size:13px; overflow-wrap:anywhere; }.err { color:oklch(45% .16 25); }button:disabled { opacity:.55; cursor:wait; }
.uses { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0 48px; }.use { padding:24px 0; border-top:1px solid var(--line); }.use h3 { font-size:14px; font-weight:500; margin:0 0 8px; }.use p { font-size:13px; color:var(--muted); margin:0 0 10px; }.use q { display:block; font-size:13px; color:var(--ink); quotes:none; }.use .save { display:inline-block; margin-top:10px; font-size:11px; color:var(--green); }
.results { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }.results th { text-align:left; font-weight:500; font-size:11px; color:var(--muted); padding:10px 12px 10px 0; border-bottom:1px solid var(--line); }.results td { padding:12px 12px 12px 0; border-bottom:1px solid var(--line); vertical-align:top; }.results td.n { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }.results th.n { text-align:right; }.results td.atr { color:var(--green); }.results caption { text-align:left; font-size:15px; font-weight:500; padding:28px 0 6px; }.findings { display:grid; grid-template-columns:1fr 1fr 1fr; gap:40px; margin-top:44px; }.findings h3 { font-size:14px; font-weight:500; margin:0 0 8px; }.findings p { font-size:13px; color:var(--muted); margin:0; }.table-scroll { overflow-x:auto; }
@media(max-width:900px) { .uses,.findings { grid-template-columns:1fr; } }
.privacy { display:grid; grid-template-columns:1fr 1fr; gap:80px; }.privacy p { font-size:13px; color:var(--muted); }.closing { padding:90px 0; text-align:center; border-top:1px solid var(--line); }.closing h2 { margin-bottom:28px; }
footer { border-top:1px solid var(--line); padding:35px 0; }.footer-row { display:flex; align-items:center; justify-content:space-between; gap:25px; }.footer-row p { color:var(--muted); font-size:11px; margin:0; }.footer-links { display:flex; gap:22px; font-size:12px; }.footer-links a { text-decoration:none; }.skip { position:absolute; left:20px; top:-80px; background:var(--ink); color:var(--surface); padding:10px; z-index:2; }.skip:focus { top:10px; }
@media(min-width:900px) { .hero { padding-top:88px; } }
@media(max-width:760px) { .wrap { width:calc(100% - 36px); }nav { min-height:78px; }.navlinks { gap:18px; }.navlinks .optional { display:none; }.navlinks .button { padding:9px 13px; }.hero { padding:52px 0 42px; }h1 { max-width:550px; margin-inline:auto; }.preview-body { grid-template-columns:1fr; }.sidebar { display:none; }.dashboard { padding:22px 20px; }.preview-top { padding:14px 20px; }.chart { gap:20px; }.chart strong { font-size:27px; }.dash-bottom { flex-wrap:wrap; }.compatibility { padding:42px 0 54px; }.apps { gap:20px 28px; font-size:17px; }.feature,.setup,.personal,.privacy { grid-template-columns:1fr; gap:32px; }.section { padding:55px 0; }.benefits { gap:24px; margin-top:40px; }.tools { column-gap:25px; }.card { padding:20px; }.closing { padding:60px 0; }.footer-row { flex-wrap:wrap; } }
@media(max-width:440px) { .navlinks>a:not(.button) { display:none; }h1 { font-size:48px; }.hero .lede { font-size:15px; }.benefits,.tools { grid-template-columns:1fr; gap:12px; }.benefits article { border-top:1px solid var(--line); padding-top:18px; }.row { grid-template-columns:1fr; }.preview-top span { font-size:10px; }.flow { padding:20px; }.tag { font-size:9px; }.chart strong small { display:block; }.chart strong { line-height:1.2; }.footer-links { gap:16px; } }

/* Neon belongs to the signal: calls to action, compressed data, and the ATR layer. */
body { overflow-x:clip; }
[hidden] { display:none!important; }
::selection { color:var(--bg); background:var(--green); }
.announcement { background:linear-gradient(90deg,transparent,oklch(88% .25 140 / .07),transparent); color:var(--muted); }
.navlinks>a:not(.button):hover,.footer-links a:hover,.caption a:hover { color:var(--green); }
.button,button { background:var(--green); border-color:var(--green); color:var(--bg); font-weight:600; box-shadow:0 0 24px oklch(88% .25 140 / .1); transition:background .2s,box-shadow .2s,transform .2s; }
.button:hover,button:hover { background:oklch(94% .14 140); box-shadow:0 0 35px oklch(88% .25 140 / .28); }
.button span { display:inline-block; transition:transform .25s; }.button:hover span { transform:translate(3px,-3px); }
.hero { position:relative; isolation:isolate; padding-bottom:56px; }
.hero::before { content:''; position:absolute; width:min(1100px,110vw); height:650px; left:50%; top:-60px; transform:translateX(-50%); background:radial-gradient(ellipse,oklch(66% .22 140 / .12),transparent 67%); pointer-events:none; z-index:-1; }
.hero::after { content:''; position:absolute; inset:0 -10px; background-image:linear-gradient(oklch(88% .25 140 / .045) 1px,transparent 1px),linear-gradient(90deg,oklch(88% .25 140 / .045) 1px,transparent 1px); background-size:64px 64px; mask-image:radial-gradient(ellipse at center,black,transparent 70%); pointer-events:none; z-index:-1; }
h1,h2 { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-weight:550; }
h1 { font-size:clamp(48px,6.7vw,88px); letter-spacing:-.065em; }h1 span { color:var(--green); text-shadow:0 0 50px oklch(88% .25 140 / .17); }
h2 { font-weight:500; }.eyebrow { color:oklch(80% .09 140); }.dot { box-shadow:0 0 12px var(--green); }
.preview { position:relative; isolation:isolate; overflow:hidden; border-color:oklch(50% .12 140 / .55); background:linear-gradient(145deg,oklch(18% .025 145),var(--surface) 50%); box-shadow:0 0 0 5px oklch(88% .25 140 / .025),0 24px 100px -40px oklch(65% .22 140 / .25); }
.preview-top { background:oklch(11% .007 145 / .65); }.preview-top strong { color:var(--green); }.preview-controls { display:flex; align-items:center; gap:14px; }.motion-toggle { padding:5px 9px; background:transparent; color:var(--muted); border:1px solid var(--line); font-size:10px; font-weight:400; box-shadow:none; white-space:nowrap; }.motion-toggle:hover { background:var(--soft); color:var(--ink); box-shadow:none; }.motion-toggle:disabled { cursor:default; }
.preview-glow { position:absolute; z-index:-1; width:380px; height:380px; top:0; left:0; border-radius:50%; background:radial-gradient(circle,oklch(88% .25 140 / .09),transparent 70%); pointer-events:none; opacity:0; }
.sidebar .selected { color:var(--green); background:linear-gradient(90deg,oklch(88% .25 140 / .1),transparent); }
.bars i { transform-origin:bottom; }.bars.reduced i { background:linear-gradient(0deg,oklch(60% .21 140),var(--green)); box-shadow:0 0 18px oklch(88% .25 140 / .17); }
.chart>div:nth-child(2)>strong { color:var(--green); }.tag { color:var(--green); border-color:oklch(88% .25 140 / .25); background:oklch(88% .25 140 / .04); }
.flow { position:relative; overflow:hidden; background:linear-gradient(145deg,var(--soft),var(--bg)); }.flow-row.atr { border-color:var(--green); box-shadow:0 0 30px oklch(88% .25 140 / .08); }.flow-row.atr strong,.flow-arrow,.number { color:var(--green); }
.card { background:linear-gradient(145deg,var(--surface),var(--bg)); }.url { color:var(--green); }input:focus-visible { border-color:var(--green); }.err { color:oklch(77% .15 25); }
.closing { position:relative; isolation:isolate; }.closing::before { content:''; position:absolute; inset:0; z-index:-1; pointer-events:none; background:radial-gradient(ellipse at bottom,oklch(88% .25 140 / .08),transparent 70%); }
.comparison { position:relative; overflow:hidden; padding-top:94px; }
.comparison::before { content:''; position:absolute; width:900px; height:620px; left:50%; top:40px; transform:translateX(-50%); background:radial-gradient(ellipse,oklch(70% .22 140 / .08),transparent 68%); pointer-events:none; }
.comparison-head { position:relative; display:grid; grid-template-columns:minmax(0,.8fr) minmax(320px,1.2fr); gap:clamp(40px,8vw,110px); align-items:end; margin-bottom:42px; }
.comparison-head h2 { max-width:520px; margin-bottom:0; }
.comparison-head p { max-width:570px; margin:0; color:var(--muted); font-size:16px; }
.comparison-visual { position:relative; padding:28px clamp(18px,4vw,48px) 22px; margin-bottom:28px; border:1px solid var(--line); background-color:oklch(12% .012 145); background-image:linear-gradient(oklch(88% .25 140 / .035) 1px,transparent 1px),linear-gradient(90deg,oklch(88% .25 140 / .035) 1px,transparent 1px); background-size:64px 64px; }
.comparison-key { display:flex; justify-content:space-between; gap:20px; color:var(--muted); font-size:11px; }
.comparison-key b { color:var(--green); font-weight:500; }
.token-stream { position:relative; width:100%; height:118px; margin:0; text-align:left; }
.token-stream svg { position:absolute; width:100%; height:100%; inset:0; overflow:visible; }.stream-track { fill:none; stroke:var(--line); stroke-width:1; }.stream-pulse { fill:none; stroke:var(--green); stroke-width:1.5; stroke-dasharray:12 260; filter:drop-shadow(0 0 4px oklch(88% .25 140 / .7)); }
.stream-label { position:absolute; top:48px; font:10px/1.5 ui-monospace,monospace; color:var(--muted); padding:4px 9px; background:oklch(12% .012 145); }.stream-label.start { left:0; }.stream-label.end { right:0; color:var(--green); }.stream-core { position:absolute; left:50%; top:31px; transform:translateX(-50%); display:grid; place-items:center; width:56px; height:56px; border:1px solid var(--green); background:var(--bg); font-size:17px; font-weight:650; color:var(--green); box-shadow:0 0 35px oklch(88% .25 140 / .13); }
.preview-body { grid-template-columns:1fr; min-height:auto; }
.dashboard { padding:clamp(26px,4vw,48px); }
.chart { gap:clamp(28px,6vw,82px); }
.metric-note { min-height:42px; max-width:36ch; margin:9px 0 0; color:var(--muted); font-size:12px!important; }
.dash-bottom { padding-top:22px; border-top:1px solid var(--line); }
@media(max-width:760px) { .hero { padding-bottom:32px; }.comparison { padding-top:60px; }.comparison-head { grid-template-columns:1fr; gap:16px; margin-bottom:30px; }.comparison-visual { padding:18px 14px 14px; }.token-stream { height:92px; }.stream-core { top:19px; width:54px; height:54px; }.stream-label { top:35px; max-width:100px; font-size:8px; line-height:1.25; padding:4px; }.comparison-key { font-size:9px; }.chart { grid-template-columns:1fr; }.metric-note { min-height:0; }.dash-heading { flex-wrap:wrap; }.preview-controls { gap:7px; }.preview-top { gap:8px; }.preview-top strong span { margin-left:5px!important; }.motion-toggle { font-size:9px; padding:4px 6px; } }

@media(prefers-reduced-motion:reduce) { html { scroll-behavior:auto; }* { transition:none!important; } }
</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="announcement"><a href="https://github.com/JacobBhatt12/atr">Free to use. Open source by design. <span aria-hidden="true">↗</span></a></div>
<header class="wrap"><nav aria-label="Main navigation"><a class="brand" href="#" aria-label="ATR home">ATR</a><div class="navlinks"><a href="#comparison">See the difference</a><a class="optional" href="#uses">Use cases</a><a class="optional" href="#tools">Tools</a><a class="optional" href="https://github.com/JacobBhatt12/atr">GitHub ↗</a><a class="button" href="#connect">Connect ATR <span aria-hidden="true">↗</span></a></div></nav></header>
<main id="main">
<section class="hero wrap" aria-labelledby="hero-title">
<p class="eyebrow"><span class="dot" aria-hidden="true"></span> A little less overhead. A lot more room.</p>
<h1 id="hero-title">Less Tokens, <span>More Usage</span></h1>
<p class="lede">ATR is a free connector for Claude, ChatGPT and other AI apps. It searches the web, reads long documents and filters large tool results so your AI sees the answer, not thousands of unnecessary tokens.</p>
<a class="button" href="#connect">Start using ATR <span aria-hidden="true">↗</span></a>
<p class="fine">Free &amp; open source · No account needed</p>
</section>
<section class="section comparison" id="comparison" aria-labelledby="comparison-title"><div class="wrap">
<div class="comparison-head"><h2 id="comparison-title">One question.<br>Two very different payloads.</h2><p>Ask which of 30 GitHub issues are about OAuth. Without ATR, your AI reads every issue and field. With ATR, it gets the two relevant issues and can retrieve anything omitted later.</p></div>
<div class="comparison-visual" aria-hidden="true"><div class="token-stream"><svg viewBox="0 0 760 105" preserveAspectRatio="none"><path class="stream-track" d="M0 15 H160 Q220 15 270 52 H380 H760 M0 52 H760 M0 90 H160 Q220 90 270 52"/><path class="stream-pulse" d="M0 15 H160 Q220 15 270 52 H380 H760"/><path class="stream-pulse" d="M0 52 H760"/><path class="stream-pulse" d="M0 90 H160 Q220 90 270 52 H380 H760"/></svg><span class="stream-label start">30 ISSUES + ALL METADATA</span><span class="stream-core">ATR</span><span class="stream-label end">2 MATCHING ISSUES</span></div><div class="comparison-key"><span>Large result enters</span><b>Only the useful part reaches your AI</b></div></div>
<div class="preview" role="img" aria-label="Measured example: asking which of 30 open GitHub issues are about OAuth. GitHub's connector returns 26,235 tokens; through ATR, 420.">
<div class="preview-top"><strong>ATR <span style="margin-left:14px">/ Token overview</span></strong><div class="preview-controls"><span>Example · GitHub issues</span><button type="button" id="motion-toggle" class="motion-toggle" aria-pressed="false" hidden>Pause motion</button></div></div><div class="preview-glow" aria-hidden="true"></div>
<div class="preview-body"><div class="dashboard"><div class="dash-heading"><h3>“Which open issues are about OAuth?”</h3><span class="tag">Measured · GitHub · 30 issues</span></div>
<div class="chart"><div><p>Without ATR</p><strong>26,235 <small>tokens</small></strong><p class="metric-note">Every issue title, body, label, author and metadata field enters the conversation.</p><div class="bars" aria-hidden="true"><i style="--h:65%"></i><i style="--h:90%"></i><i style="--h:76%"></i><i style="--h:100%"></i><i style="--h:86%"></i><i style="--h:70%"></i><i style="--h:95%"></i><i style="--h:80%"></i><i style="--h:92%"></i><i style="--h:72%"></i><i style="--h:88%"></i><i style="--h:96%"></i></div></div>
<div><p>With ATR</p><strong>420 <small>tokens</small></strong><p class="metric-note">Only the issues that answer the question enter the conversation. The full result stays retrievable.</p><div class="bars reduced" aria-hidden="true"><i style="--h:3%"></i><i style="--h:4%"></i><i style="--h:3%"></i><i style="--h:5%"></i><i style="--h:3%"></i><i style="--h:4%"></i><i style="--h:3%"></i><i style="--h:4%"></i><i style="--h:5%"></i><i style="--h:3%"></i><i style="--h:4%"></i><i style="--h:3%"></i></div></div></div>
<div class="dash-bottom"><span>The matching issues kept. The other 28 left out, but saved if asked for.</span><b>98% fewer tokens in this result ↗</b></div></div></div></div>
<p class="caption">Measured November 6, 2026 on GitHub's official MCP server. Savings vary by task. <a href="#results">See all results ↓</a></p>
</div></section>
<section class="compatibility wrap" aria-label="Compatible apps"><p>At home with the AI tools you already use</p><div class="apps"><span><svg class="app-logo" xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 256 257" fill="currentColor" aria-hidden="true" focusable="false"><path fill="#D97757" d="m50.228 170.321 50.357-28.257.843-2.463-.843-1.361h-2.462l-8.426-.518-28.775-.778-24.952-1.037-24.175-1.296-6.092-1.297L0 125.796l.583-3.759 5.12-3.434 7.324.648 16.202 1.101 24.304 1.685 17.629 1.037 26.118 2.722h4.148l.583-1.685-1.426-1.037-1.101-1.037-25.147-17.045-27.22-18.017-14.258-10.37-7.713-5.25-3.888-4.925-1.685-10.758 7-7.713 9.397.649 2.398.648 9.527 7.323 20.35 15.75L94.817 91.9l3.889 3.24 1.555-1.102.195-.777-1.75-2.917-14.453-26.118-15.425-26.572-6.87-11.018-1.814-6.61c-.648-2.723-1.102-4.991-1.102-7.778l7.972-10.823L71.42 0 82.05 1.426l4.472 3.888 6.61 15.101 10.694 23.786 16.591 32.34 4.861 9.592 2.592 8.879.973 2.722h1.685v-1.556l1.36-18.211 2.528-22.36 2.463-28.776.843-8.1 4.018-9.722 7.971-5.25 6.222 2.981 5.12 7.324-.713 4.73-3.046 19.768-5.962 30.98-3.889 20.739h2.268l2.593-2.593 10.499-13.934 17.628-22.036 7.778-8.749 9.073-9.657 5.833-4.601h11.018l8.1 12.055-3.628 12.443-11.342 14.388-9.398 12.184-13.48 18.147-8.426 14.518.778 1.166 2.01-.194 30.46-6.481 16.462-2.982 19.637-3.37 8.88 4.148.971 4.213-3.5 8.62-20.998 5.184-24.628 4.926-36.682 8.685-.454.324.519.648 16.526 1.555 7.065.389h17.304l32.21 2.398 8.426 5.574 5.055 6.805-.843 5.184-12.962 6.611-17.498-4.148-40.83-9.721-14-3.5h-1.944v1.167l11.666 11.406 21.387 19.314 26.767 24.887 1.36 6.157-3.434 4.86-3.63-.518-23.526-17.693-9.073-7.972-20.545-17.304h-1.36v1.814l4.73 6.935 25.017 37.59 1.296 11.536-1.814 3.76-6.481 2.268-7.13-1.297-14.647-20.544-15.1-23.138-12.185-20.739-1.49.843-7.194 77.448-3.37 3.953-7.778 2.981-6.48-4.925-3.436-7.972 3.435-15.749 4.148-20.544 3.37-16.333 3.046-20.285 1.815-6.74-.13-.454-1.49.194-15.295 20.999-23.267 31.433-18.406 19.702-4.407 1.75-7.648-3.954.713-7.064 4.277-6.286 25.47-32.405 15.36-20.092 9.917-11.6-.065-1.686h-.583L44.07 198.125l-12.055 1.555-5.185-4.86.648-7.972 2.463-2.593 20.35-13.999-.064.065Z"/></svg>Claude</span><span><svg class="app-logo" xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="100 100 410 410" fill="currentColor" aria-hidden="true" focusable="false"><path fill-rule="evenodd" clip-rule="evenodd" d="M252.794 108.802C289.191 99.0484 326.265 110.305 351.148 135.135C385.113 126.072 422.85 134.862 449.492 161.505C476.136 188.149 484.925 225.888 475.862 259.85V259.854C500.696 284.735 511.95 321.81 502.198 358.207C492.447 394.602 464.161 421.084 430.215 430.217C421.083 464.162 394.603 492.448 358.206 502.199C321.812 511.951 284.734 500.693 259.852 475.864C225.887 484.927 188.15 476.137 161.507 449.495C134.864 422.851 126.073 385.111 135.136 351.149C110.304 326.266 99.0496 289.192 108.801 252.795C118.552 216.4 146.84 189.918 180.784 180.785C189.917 146.841 216.396 118.553 252.794 108.802ZM374.292 407.145C374.292 411.271 372.092 415.086 368.517 417.148L283.723 466.102C302.487 480.585 327.555 486.459 352.217 479.852C386.997 470.532 410.068 439.312 410.555 405.006V317.717C410.555 315.08 409.125 312.621 406.843 311.303L374.292 292.509V407.145ZM251.868 415.897C248.296 417.959 243.893 417.959 240.317 415.897L155.526 366.942C152.366 390.436 159.811 415.08 177.866 433.136H177.863C203.325 458.594 241.896 462.962 271.85 446.232L347.449 402.586C349.735 401.268 351.148 398.8 351.148 396.163V358.579L251.868 415.897ZM368.602 220.628C366.319 219.309 363.474 219.318 361.191 220.637L328.641 239.431L427.921 296.749C431.496 298.811 433.697 302.627 433.697 306.752V404.661C455.622 395.654 473.244 376.881 479.851 352.218C489.169 317.442 473.668 281.85 444.201 264.274L368.602 220.628ZM177.303 206.34C155.377 215.348 137.756 234.122 131.148 258.783C121.832 293.561 137.331 329.153 166.799 346.727L242.398 390.373C244.68 391.692 247.525 391.684 249.807 390.366L282.357 371.572L183.078 314.253C179.504 312.189 177.303 308.375 177.303 304.251V206.34ZM259.849 279.145V331.858L305.5 358.213L351.15 331.858V279.145L305.5 252.789L259.849 279.145ZM327.276 144.9C308.512 130.418 283.445 124.543 258.782 131.15C224.002 140.471 200.931 171.691 200.445 205.995V293.286C200.445 295.923 201.875 298.381 204.158 299.7L236.707 318.493V203.856C236.707 199.731 238.909 195.916 242.483 193.853L327.276 144.9ZM433.137 177.867C407.675 152.407 369.103 148.038 339.149 164.769L263.55 208.415C261.265 209.734 259.852 212.202 259.852 214.838V252.423L359.132 195.105C362.703 193.041 367.108 193.041 370.682 195.105L455.473 244.06C458.635 220.567 451.189 195.922 433.135 177.867H433.137Z" fill="white"/></svg>ChatGPT</span><span><svg class="app-logo" xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false"><path clip-rule="evenodd" d="M8.086.457a6.105 6.105 0 013.046-.415c1.333.153 2.521.72 3.564 1.7a.117.117 0 00.107.029c1.408-.346 2.762-.224 4.061.366l.063.03.154.076c1.357.703 2.33 1.77 2.918 3.198.278.679.418 1.388.421 2.126a5.655 5.655 0 01-.18 1.631.167.167 0 00.04.155 5.982 5.982 0 011.578 2.891c.385 1.901-.01 3.615-1.183 5.14l-.182.22a6.063 6.063 0 01-2.934 1.851.162.162 0 00-.108.102c-.255.736-.511 1.364-.987 1.992-1.199 1.582-2.962 2.462-4.948 2.451-1.583-.008-2.986-.587-4.21-1.736a.145.145 0 00-.14-.032c-.518.167-1.04.191-1.604.185a5.924 5.924 0 01-2.595-.622 6.058 6.058 0 01-2.146-1.781c-.203-.269-.404-.522-.551-.821a7.74 7.74 0 01-.495-1.283 6.11 6.11 0 01-.017-3.064.166.166 0 00.008-.074.115.115 0 00-.037-.064 5.958 5.958 0 01-1.38-2.202 5.196 5.196 0 01-.333-1.589 6.915 6.915 0 01.188-2.132c.45-1.484 1.309-2.648 2.577-3.493.282-.188.55-.334.802-.438.286-.12.573-.22.861-.304a.129.129 0 00.087-.087A6.016 6.016 0 015.635 2.31C6.315 1.464 7.132.846 8.086.457zm-.804 7.85a.848.848 0 00-1.473.842l1.694 2.965-1.688 2.848a.849.849 0 001.46.864l1.94-3.272a.849.849 0 00.007-.854l-1.94-3.393zm5.446 6.24a.849.849 0 000 1.695h4.848a.849.849 0 000-1.696h-4.848z" fill-rule="evenodd"/></svg>Codex</span><span><svg class="app-logo" xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 466.73 532.09" fill="currentColor" aria-hidden="true" focusable="false"><path d="M457.43,125.94L244.42,2.96c-6.84-3.95-15.28-3.95-22.12,0L9.3,125.94c-5.75,3.32-9.3,9.46-9.3,16.11v247.99c0,6.65,3.55,12.79,9.3,16.11l213.01,122.98c6.84,3.95,15.28,3.95,22.12,0l213.01-122.98c5.75-3.32,9.3-9.46,9.3-16.11v-247.99c0-6.65-3.55-12.79-9.3-16.11h-.01ZM444.05,151.99l-205.63,356.16c-1.39,2.4-5.06,1.42-5.06-1.36v-233.21c0-4.66-2.49-8.97-6.53-11.31L24.87,145.67c-2.4-1.39-1.42-5.06,1.36-5.06h411.26c5.84,0,9.49,6.33,6.57,11.39h-.01Z" fill="#edecec"/></svg>Cursor</span><small>and other MCP clients</small></div></section>
<section class="section wrap" id="how-it-works"><div class="feature"><div><h2>More signal.<br>Less baggage.</h2><p>AI apps measure reading in <b>tokens</b>, roughly ¾ of a word each. Everything your AI reads counts: your messages, its replies, and every web page or tool result it pulls in. It re-reads all of it with every new step. A single list from GitHub can be 26,000 tokens, which uses up a chat's room and your plan's limits fast.</p><p>ATR sits between your AI and the places it reads from. It passes along the parts that answer your question, quoted exactly, and keeps the rest on hand in case your AI asks for it.</p><a class="text-link" href="#tools">Explore what ATR does <span aria-hidden="true">↗</span></a></div><div class="flow" aria-label="How ATR connects your AI to tools"><div class="flow-row"><span>Your AI app</span><small>Claude, ChatGPT, Codex…</small></div><div class="flow-arrow" aria-hidden="true">↓</div><div class="flow-row atr"><strong>ATR</strong><small>Find → Run → Reduce</small></div><div class="flow-arrow" aria-hidden="true">↓</div><div class="flow-row"><span>Your tools &amp; data</span><small>Web · Documents · MCP</small></div></div></div>
<div class="benefits"><article><span class="number">01 / Discover</span><h3>A smaller tool list</h3><p>Apps like GitHub come with dozens of tools, each with its own instructions. Behind ATR, your AI sees 7 small tools and looks up the one it needs: 917 tokens instead of 32,563 for GitHub.</p></article><article><span class="number">02 / Reduce</span><h3>Only the part you asked about</h3><p>Your AI tells ATR what it is looking for. ATR returns the matching sentences, rows or lines, word for word, from the live page or app.</p></article><article><span class="number">03 / Retrieve</span><h3>Nothing is lost</h3><p>The full result is kept for a while. If your AI needs a detail that was left out, it can search or filter it without loading everything again.</p></article></div></section>
<section class="section wrap" id="uses"><h2>What it's good for</h2><p class="section-intro">ATR helps most when your AI would otherwise read a lot to answer a little. Here's what that looks like in a chat.</p><div class="uses">
<article class="use"><h3>Questions about your GitHub (or other connected apps)</h3><q>“Which open issues are about login bugs?”</q><p>Instead of every issue in full, your AI gets the few that match.</p><span class="save">26,235 → 420 tokens</span></article>
<article class="use"><h3>Reviewing pull requests</h3><q>“Any open PRs that touch the payments code?”</q><p>Only the matching pull requests come back, not 30 full descriptions.</p><span class="save">28,284 → 392 tokens</span></article>
<article class="use"><h3>Pages that change</h3><q>“What's on the Hacker News front page right now?”</q><p>ATR reads the live page. In our tests, Claude's built-in reader returned an old copy every time.</p><span class="save">Right every time in our tests</span></article>
<article class="use"><h3>Exact numbers and versions</h3><q>“What Python version does this library need?”</q><p>ATR quotes the page word for word, so numbers aren't paraphrased or mixed up with an older version.</p><span class="save">Quoted, not summarized</span></article>
<article class="use"><h3>One question about a long document</h3><q>“In this PDF, what scores did the model get?”</q><p>Works with PDFs, Word, PowerPoint and Excel links. You get the matching sentences instead of 40 pages.</p><span class="save">About 250 tokens per answer</span></article>
<article class="use"><h3>Coding with an AI agent</h3><q>“Run the tests and fix what fails.”</q><p>With ATR installed on your computer, test runs keep the failures and drop hundreds of “passed” lines.</p><span class="save">6,685 → 675 tokens</span></article>
</div></section>
<section class="section wrap" id="results"><h2>Measured results</h2><p class="section-intro">We tested ATR in real claude.ai chats and against GitHub's official MCP server on November 6, 2026. Tokens are estimated as characters ÷ 4. The comparison includes where ATR saves tokens, where it improves freshness, and where it costs more.</p>
<div class="table-scroll"><table class="results"><caption>Data from a connected app: GitHub's official MCP server (46 tools)</caption><thead><tr><th>What your AI has to read</th><th class="n">Without ATR</th><th class="n">With ATR</th></tr></thead><tbody>
<tr><td>Instructions for every tool (in apps that load them all)</td><td class="n">32,563</td><td class="n atr">917</td></tr>
<tr><td>30 open issues: “which are about OAuth?”</td><td class="n">26,235</td><td class="n atr">420</td></tr>
<tr><td>30 open pull requests: “which touch streamable HTTP?”</td><td class="n">28,284</td><td class="n atr">392</td></tr>
<tr><td>Repository search: “which have the most stars?”</td><td class="n">4,232</td><td class="n atr">369</td></tr>
<tr><td>A README: “what's the install command?”</td><td class="n">1,160</td><td class="n atr">213</td></tr>
</tbody></table></div>
<div class="table-scroll"><table class="results"><caption>Reading web pages in claude.ai: four tasks (Wikipedia, a PDF, Hacker News, a GitHub README)</caption><thead><tr><th></th><th class="n">Claude's built-in reader</th><th class="n">ATR</th></tr></thead><tbody>
<tr><td>Right answers</td><td class="n">Wrong on every Hacker News and README question (0 of 8)</td><td class="n atr">28 of 28</td></tr>
<tr><td>Tokens for all four tasks</td><td class="n">1,698</td><td class="n atr">4,144</td></tr>
</tbody></table></div>
<div class="findings"><div><h3>Big savings on app data</h3><p>When your AI pulls lists, issues, records or files from a connected app, ATR cut what it had to read by 82–99%.</p></div><div><h3>Fresher, exact answers from the web</h3><p>ATR read the live page every time. Claude's built-in reader is cheaper on simple pages (about 2.4×), but it served old copies of pages that change.</p></div><div><h3>You may need to ask for it</h3><p>In claude.ai, Claude uses its own reader unless told otherwise. One line in your preferences fixes that (see setup below).</p></div></div>
<p class="caption"><a href="https://github.com/JacobBhatt12/atr/tree/main/benchmarks">How we tested, with every chat and number ↗</a></p></section>
<section class="section wrap" id="connect"><div class="setup"><div><h2>Set it up in two minutes</h2><p class="section-intro">A <b>connector</b> is an add-on that gives your chatbot new tools. Copy this link, add it as a connector in your app, and ATR is ready in your next chat.</p><div class="copy"><div class="url" id="base" tabindex="0" aria-label="ATR connector URL">{base}/mcp</div><button type="button" data-copy="base">Copy</button></div><p class="muted">No account or sign-in. Choose “No authentication” when asked.</p></div><div class="instructions"><details open><summary>Claude &amp; Claude Desktop</summary><ol><li>Open Settings → Connectors → <b>Add custom connector</b>.</li><li>Name it <b>ATR</b>, paste the connector link, and click Add.</li><li>Open a new chat. Make sure ATR is switched on under <b>+ → Connectors</b>.</li><li>Claude will use its built-in reader unless you ask for ATR. To make ATR the default, add this line to Settings → Profile → personal preferences (or a Project's instructions): <i>“When I ask you to search the web or read a link, use atr_search and atr_fetch instead of the built-in web tools.”</i> Or skip the setup and ask your question and add “use atr”, like <i>“What is the latest on the MCP spec? Use atr.”</i></li></ol></details><details><summary>ChatGPT</summary><ol><li>Open Settings → Apps &amp; Connectors → Advanced and enable <b>Developer mode</b>.</li><li>Choose Create, name it <b>ATR</b>, and paste the link.</li><li>Set Authentication to <b>No authentication</b>.</li></ol></details><details><summary>Other MCP apps</summary><p>Add the link as a remote MCP server using Streamable HTTP. Available connector options depend on your app and plan.</p></details><details><summary>Local coding tools</summary><p>For local files, shell commands and coding tools, install ATR on your computer:</p><p><code>pip install atr-mcp<br>atr setup</code></p><p><a href="https://github.com/JacobBhatt12/atr#get-started">Read the local setup guide ↗</a></p></details></div></div></section>
<section class="section wrap" id="tools"><h2>What your chatbot gets</h2><p class="section-intro">Seven small tools. You don't need to learn them: your AI picks the right one. These are the names you'll see in the chat.</p><div class="tools"><article class="tool"><h3>Read the web &amp; documents</h3><p><code>atr_fetch</code> reads a link (web page, PDF, Word, PowerPoint, Excel or JSON) and returns the sentences that answer the question.</p></article><article class="tool"><h3>Search the web</h3><p><code>atr_search</code> asks DuckDuckGo, Bing and Brave at once and returns one short list: title, link and one line each. Five results are about 300 tokens.</p></article><article class="tool"><h3>Inspect a website</h3><p><code>atr_site</code> shows a site's title, description, language, headings and link counts, or searches its public sitemap and navigation.</p></article><article class="tool"><h3>Find local places</h3><p><code>atr_places</code> finds cafes, restaurants, stores, services and other places using free OpenStreetMap data. No API key needed.</p></article><article class="tool"><h3>Check the weather</h3><p><code>atr_weather</code> returns current conditions and up to seven forecast days for a city, postal code or address.</p></article><article class="tool"><h3>Look up what was left out</h3><p><code>atr_query</code> searches or filters a full result ATR kept, so your AI can get one more detail without reading everything again.</p></article><article class="tool"><h3>See what you saved</h3><p><code>atr_stats</code> shows how many tokens ATR saved in your chat. Ask “how much did ATR save?”</p></article></div></section>
<section class="section wrap" id="personal"><div class="personal"><div><h2>Put your other apps behind ATR</h2><p>This is where the biggest savings are. If you use an online MCP server that takes an API key or needs no sign-in (GitHub, DeepWiki and many others), add it here. You get one personal connector link. Use it <i>instead of</i> the original connector, and your AI gets the same tools with much smaller results.</p><p class="muted">Not yet supported: apps you sign into through your chatbot, like Canva, Notion, Figma or Gmail.</p><p class="muted">Your personal link contains encrypted credentials. Keep it private: anyone with the link can use those servers.</p></div><div>{personal}</div></div></section>
<section class="section wrap"><div class="privacy"><div><h2>Less to store.<br>Less to worry about.</h2><p>ATR keeps no database and doesn't record what you read. Large results are held in memory briefly for follow-up queries, then dropped.</p></div><div class="instructions"><details><summary>Where are my server keys stored?</summary><p>In your personal link, encrypted. They aren't stored in a database. Treat the link like a password and only add it to apps you trust.</p></details><details><summary>Can ATR work with email, Word, PowerPoint or Excel?</summary><p><code>atr_fetch</code> can read Word, PowerPoint and Excel files from a public link. Sending email or editing cloud files requires access to your account, so those actions need a connected email or Microsoft MCP server. The public ATR connector never asks for those credentials.</p></details><details><summary>Can the hosted connector read my local files?</summary><p>No. It reads public websites and online documents. It cannot reach your computer, local files or private networks.</p></details><details><summary>Does ATR change my usage limits?</summary><p>ATR reduces tool overhead. It doesn't change your plan's limits, compress your conversation history or shorten the AI's own replies. Savings depend on the tools and tasks you use.</p></details><details><summary>When doesn't ATR help?</summary><p>Short, simple web pages in claude.ai: Claude's built-in reader uses fewer tokens there. Also short answers from apps, where there's little to trim. ATR helps most with long pages, documents, pages that change, and large results from connected apps.</p></details><details><summary>What about tools that make changes?</summary><p>Read operations and changes use separate tools. Changes go through atr_call so your chatbot can request approval before running them.</p></details></div></div></section>
<section class="closing wrap"><h2>Give your AI a little more room.</h2><a class="button" href="#connect">Connect ATR <span aria-hidden="true">↗</span></a></section>
</main>
<footer><div class="wrap footer-row"><a class="brand" href="#" aria-label="ATR home">ATR</a><p>Automation for Token Reduction · Open source, MIT licensed.</p><div class="footer-links"><a href="https://github.com/JacobBhatt12/atr">GitHub ↗</a><a href="https://github.com/JacobBhatt12/atr#get-started" >Documentation ↗</a></div></div></footer>
<script defer src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js" integrity="sha384-HOvlOYPIs/zjoIkWUGXkVmXsjr8GuZLV+Q+rcPwmJOVZVpvTSXQChiN4t9Euv9Vc" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/ScrollTrigger.min.js" integrity="sha384-P8VzCVnT9NBUkMrpcIZrJbA7EBjJvh/fJS6PmP+4nLIM284DtsImIv8D0fFjIkeh" crossorigin="anonymous"></script>
<script>
  document.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-copy]'); if (!b) return;
    const text = document.getElementById(b.dataset.copy).textContent.trim();
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; setTimeout(() => b.textContent = 'Copy', 1500); }
    catch { b.textContent = 'Select and copy'; }
  });
  const servers = document.getElementById('servers');
  function addServer() {
    if (!servers) return;
    const n = servers.children.length + 1;
    const d = document.createElement('div'); d.className = 'server';
    d.innerHTML = `
      <div class="row"><div><label for="name-${n}">Name</label><input id="name-${n}" name="name" placeholder="github" required></div>
        <div><label for="url-${n}">Server URL (https)</label><input id="url-${n}" name="url" type="url" placeholder="https://api.example.com/mcp" required></div></div>
      <div class="row hdr"><div><label for="hname-${n}">Header name (optional)</label><input id="hname-${n}" name="hname" placeholder="Authorization"></div>
        <div><label for="hvalue-${n}">Header value</label><input id="hvalue-${n}" name="hvalue" type="password" autocomplete="off" placeholder="Bearer sk-…"></div></div>`;
    servers.appendChild(d);
  }
  addServer();
  document.getElementById('add')?.addEventListener('click', addServer);
  document.getElementById('form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const submit = e.target.querySelector('[type=submit]');
    submit.disabled = true;
    const out = document.getElementById('result'); out.textContent = 'Checking your servers…';
    const list = [...servers.children].map(d => {
      const v = (n) => d.querySelector(`[name=${n}]`).value.trim();
      return { name: v('name'), url: v('url'), headers: v('hname') ? { [v('hname')]: v('hvalue') } : {} };
    }).filter(s => s.name || s.url);
    try {
      const r = await fetch('/api/link', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ servers: list }) });
      const j = await r.json();
      if (!r.ok) { out.innerHTML = '<p class="err"></p>'; out.firstChild.textContent = j.error || 'Something went wrong.'; return; }
      out.innerHTML = '<p>Your personal link (add it the same way as above, and keep it private):</p>' +
        '<div class="copy"><div class="url" id="mine"></div><button type="button" data-copy="mine">Copy</button></div><ul id="report"></ul>';
      document.getElementById('mine').textContent = j.url;
      for (const [name, s] of Object.entries(j.servers)) {
        const li = document.createElement('li');
        li.textContent = s.tools ? `${name}: connected, ${s.tools} tools` : `${name}: ${s.error}`;
        if (!s.tools) li.className = 'err';
        document.getElementById('report').appendChild(li);
      }
    } catch { out.innerHTML = '<p class="err">Couldn\\'t reach the server. Try again.</p>'; }
    finally { submit.disabled = false; }
  });
</script>

<script>
  window.addEventListener('DOMContentLoaded', () => {
    if (!window.gsap || !window.ScrollTrigger) return;
    gsap.registerPlugin(ScrollTrigger);
    const toggle = document.getElementById('motion-toggle');
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let paused = false;
    let media;
    function startMotion() {
      media?.revert();
      toggle.hidden = preference.matches;
      if (paused || preference.matches) return;
      media = gsap.matchMedia();
      media.add('(prefers-reduced-motion: no-preference)', () => {
        const entrance = gsap.timeline({defaults:{ease:'power3.out',duration:0.95}});
        entrance.from('.hero .eyebrow', {y:14,opacity:0})
          .from('.hero h1', {y:30,opacity:0},0.1)
          .from('.hero .lede', {y:20,opacity:0},0.24)
          .from('.hero > .button, .hero .fine', {y:14,opacity:0,stagger:0.1},0.36);
        const stream = gsap.to('.stream-pulse', {strokeDashoffset:-1088,duration:7,repeat:-1,ease:'none',stagger:0.7,paused:true});
        ScrollTrigger.create({trigger:'.token-stream',start:'top bottom',end:'bottom top',onToggle:self => self.isActive ? stream.play() : stream.pause()});
        gsap.from('.bars i', {scaleY:0,duration:1.2,stagger:0.035,ease:'power3.out',scrollTrigger:{trigger:'.chart',start:'top 92%',once:true}});
        const compressed = gsap.to('.bars.reduced i', {opacity:0.55,duration:1.4,stagger:0.08,repeat:-1,yoyo:true,ease:'sine.inOut',paused:true});
        ScrollTrigger.create({trigger:'.chart',start:'top bottom',end:'bottom top',onToggle:self => self.isActive ? compressed.play() : compressed.pause()});
        gsap.utils.toArray('.section, .closing, .compatibility').forEach(section => {
          const children = section.querySelectorAll('h2, .section-intro, .comparison-head > p, .comparison-visual, .preview, .feature > div, .benefits article, .setup > div, .tool, .personal > div, .privacy > div, .apps > *, .closing .button');
          // Animate direct compositions once; nested elements keep their own visual hierarchy.
          const targets = [...children].filter(el => ![...children].some(parent => parent !== el && parent.contains(el)));
          if (targets.length) gsap.from(targets,{y:28,opacity:0,duration:0.85,stagger:0.09,ease:'power3.out',scrollTrigger:{trigger:section,start:'top 88%',once:true}});
        });
        const flow = gsap.timeline({repeat:-1,repeatDelay:1,paused:true});
        flow.fromTo('.flow-arrow',{opacity:0.25,y:-5},{opacity:1,y:3,duration:0.8,stagger:0.35,ease:'power2.out'})
          .to('.flow-row.atr',{boxShadow:'0 0 34px rgba(110,255,85,0.22)',duration:0.7},0.3)
          .to('.flow-row.atr',{boxShadow:'0 0 30px rgba(110,255,85,0.04)',duration:0.8});
        ScrollTrigger.create({trigger:'.flow',start:'top bottom',end:'bottom top',onToggle:self => self.isActive ? flow.play() : flow.pause()});
        // Only fine pointers receive the light-following interaction.
        const preview = document.querySelector('.preview');
        const glow = preview.querySelector('.preview-glow');
        const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
        const moveX = gsap.quickTo(glow,'x',{duration:0.65,ease:'power3.out'});
        const moveY = gsap.quickTo(glow,'y',{duration:0.65,ease:'power3.out'});
        function move(event) {
          const rect = preview.getBoundingClientRect();
          moveX(event.clientX-rect.left-190); moveY(event.clientY-rect.top-190);
          gsap.to(glow,{opacity:1,duration:0.3,overwrite:'auto'});
        }
        function leave() { gsap.to(glow,{opacity:0,duration:0.5,overwrite:'auto'}); }
        if (finePointer) { preview.addEventListener('pointermove',move); preview.addEventListener('pointerleave',leave); }
        // Native disclosure changes can shift the scroll positions below them.
        const refresh = () => ScrollTrigger.refresh();
        document.querySelectorAll('details').forEach(detail => detail.addEventListener('toggle',refresh));
        const visibility = () => {
          if (document.hidden) { stream.pause(); compressed.pause(); flow.pause(); }
          else { ScrollTrigger.refresh(); ScrollTrigger.getAll().forEach(trigger => { if (trigger.vars.onToggle) trigger.vars.onToggle(trigger); }); }
        };
        document.addEventListener('visibilitychange',visibility);
        return () => {
          preview.removeEventListener('pointermove',move); preview.removeEventListener('pointerleave',leave);
          document.querySelectorAll('details').forEach(detail => detail.removeEventListener('toggle',refresh));
          document.removeEventListener('visibilitychange',visibility);
        };
      });
    }
    toggle.addEventListener('click', () => {
      paused = !paused;
      toggle.textContent = paused ? 'Resume motion' : 'Pause motion';
      toggle.setAttribute('aria-pressed',String(paused));
      startMotion();
    });
    preference.addEventListener('change',startMotion);
    startMotion();
  });
</script>
</body>
</html>
"""

FORM = """<form class="card" id="form">
    <div id="servers"></div>
    <div class="actions"><button type="button" class="ghost" id="add">+ Another server</button><button type="submit">Make my link</button></div>
    <div id="result" aria-live="polite"></div>
  </form>
  <p class="muted">Servers that need you to sign in through a browser (OAuth) can't go behind ATR yet. Keep those as their own connectors.</p>"""

OFF = '<p class="muted">Personal links are turned off on this server.</p>'


def render(base: str, personal: bool) -> str:
    return PAGE.replace("{base}", html.escape(base)).replace("{personal}", FORM if personal else OFF)
