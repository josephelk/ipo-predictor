"""Gaya (CSS) aplikasi. Kelas lama (res-grid, fi-*, feat, hero) dari versi sebelumnya; menu kiri dan kartu info baru."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--bg:#171717;--card:#222;--card2:#1c1c1c;--line:#343434;--maroon:#4e1a17;--maroon2:#62231d;
      --text:#f2f2f2;--muted:#9b9b9b;--green:#b6dcae;--red:#e9a39b;--blue:#6ea8ff}
html,body,.stApp{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif}
#MainMenu,footer,[data-testid="stMainMenu"],[data-testid="stAppDeployButton"]{display:none}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1120px;width:calc(100% - 2.4rem);margin:3.4rem auto 2rem;padding:0 2.5rem 2.5rem 2.5rem;background:var(--card);
  border:1px solid var(--line);border-radius:14px}
.app-header{margin:0 -2.5rem 2rem -2.5rem;padding:1.15rem 2.5rem;background:var(--maroon);
  border-bottom:1px solid #00000055;font-weight:600;font-size:1rem;border-radius:13px 13px 0 0}
h1,h2,h3{color:var(--text)}
.hero{text-align:center;padding:2rem 1rem 0.5rem}
.hero h1{font-size:2.3rem;font-weight:700;line-height:1.2;margin:0 0 1rem}
.hero p{color:var(--muted);font-size:1rem;max-width:640px;margin:0 auto 1.4rem}
.feat{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.2rem;height:100%}
.feat svg{width:20px;height:20px;stroke:var(--blue);fill:none;stroke-width:1.8;margin-bottom:.9rem}
.feat b{display:block;font-size:.95rem;margin-bottom:.35rem}
.feat span{color:var(--muted);font-size:.78rem;line-height:1.4}
.stButton>button,.stDownloadButton>button{background:var(--maroon);border:1px solid #6b2b25;color:var(--text);
  border-radius:8px;padding:.7rem 1rem;font-weight:600;width:100%}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--maroon2);border-color:#8a3a32;color:#fff}
div[data-baseweb="input"],div[data-baseweb="select"]>div,div[data-baseweb="base-input"]{
  background:#1a1a1a!important;border-color:#2f2f2f!important;border-radius:6px!important}
[data-testid="stNumberInputStepUp"],[data-testid="stNumberInputStepDown"]{display:none}
[data-testid="stWidgetLabel"] p{font-size:.78rem;color:#d4d4d4;font-weight:500}
[data-testid="stFileUploaderDropzone"]{background:#1a1a1a;border:1px dashed #4a4a4a;border-radius:10px}
[data-testid="stExpander"]{background:var(--card2);border:1px solid var(--line);border-radius:10px}
.res-grid{display:grid;grid-template-columns:1.15fr 1fr 1fr;grid-template-rows:1fr 1fr;gap:1rem;margin:1rem 0 2rem}
.res-main{grid-row:1/3;border-radius:10px;padding:1.5rem;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;border:1px solid}
.res-main.ok{background:#26391f;border-color:#3d5c33}.res-main.no{background:#3a2220;border-color:#6a3b36}
.res-main .lbl{font-size:.7rem;letter-spacing:.12em;color:var(--muted)}
.res-main .big{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:600;margin:.5rem 0}
.res-main.ok .big{color:var(--green)}.res-main.no .big{color:var(--red)}
.res-main .sub{font-size:.8rem;color:#c9c9c9}
.res-stat{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:1rem;
  text-align:center;display:flex;flex-direction:column;justify-content:center}
.res-stat .lbl{font-size:.7rem;color:var(--muted);font-family:'JetBrains Mono',monospace}
.res-stat .val{font-family:'JetBrains Mono',monospace;font-size:1.35rem;font-weight:600;margin-top:.35rem}
.res-stat .val.g{color:var(--green)}.res-stat .val.r{color:var(--red)}
.fi-title{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:600;margin:.5rem 0 1rem}
.fi-row{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.75rem;margin-top:.9rem}
.fi-row .n{color:var(--blue)}
.fi-bar{height:6px;background:#303030;border-radius:4px;margin-top:.35rem;overflow:hidden}
.fi-bar>div{height:100%;background:#2f4f8f;border-radius:4px}
.cap{color:var(--muted);font-size:.75rem}
@media(max-width:700px){.res-grid{grid-template-columns:1fr 1fr}.res-main{grid-column:1/3;grid-row:auto}
  .block-container{padding:0 1.2rem 1.5rem}.app-header{margin:0 -1.2rem 1.5rem;padding:1rem 1.2rem}}

/* ---- menu kiri ---- */
[data-testid="stSidebar"]{background:var(--card2);border-right:1px solid var(--line)}
[data-testid="stSidebarNav"]::before{content:"IPO Stock Performance Predictor";display:block;padding:.5rem .9rem 1rem;
  font-weight:700;font-size:.95rem;line-height:1.3;color:var(--text)}
[data-testid="stSidebarNavLink"]{border-radius:8px;margin:2px 0}
[data-testid="stSidebarNavLink"]:hover{background:#2b2b2b}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:var(--maroon)!important;color:#fff!important}
[data-testid="stSidebarUserContent"] [data-testid="stCaptionContainer"]{padding:0 .9rem;color:var(--muted)}
/* ---- halaman informasi ---- */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:.8rem}
.kpi{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem}
.kpi .k-lbl{font-size:.78rem;color:var(--muted)}
.kpi .k-val{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:600;margin:.4rem 0 .3rem;line-height:1.2;overflow-wrap:anywhere}
.kpi .k-ket{font-size:.76rem;color:var(--muted);line-height:1.4}
.langkah{display:flex;gap:1rem;margin:.9rem 0;align-items:flex-start}
.langkah .no{width:1.9rem;height:1.9rem;flex:none;border-radius:50%;background:var(--maroon2);display:flex;align-items:center;
  justify-content:center;font-family:'JetBrains Mono',monospace;font-weight:600;font-size:.85rem}
.langkah span{color:#c9c9c9;font-size:.9rem;line-height:1.5}
.istilah{padding:.75rem 0;border-bottom:1px solid var(--line)}
.istilah b{display:block;margin-bottom:.2rem}
.istilah span{color:#c9c9c9;font-size:.9rem;line-height:1.5}
@media(max-width:900px){.kpi-grid{grid-template-columns:1fr 1fr}}
@media(max-width:700px){.block-container{margin-top:3.2rem}}
</style>
"""
