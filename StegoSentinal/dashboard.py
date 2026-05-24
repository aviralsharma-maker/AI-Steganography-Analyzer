#!/usr/bin/env python3
#  STEGOSENTINEL DASHBOARD — Fully Integrated Version
#
# This dashboard owns the model, runs batch evaluation
# in a background thread, and serves live results via Flask.
#
# Run:  python dashboard.py
# Open: http://localhost:5000
# Then click "Run Batch" in the UI.

import os
import json
import threading
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '0'


#   CONFIG

IMG_SIZE   = 256
N_CHANNELS = 18

BASE         = '/home/pranj'
MODEL_PATH   = os.path.join(BASE, 'steganalysis', 'best_model.keras')
ALERT_LOG    = os.path.join(BASE, 'steganalysis', 'alerts.json')
SUMMARY_PATH = os.path.join(BASE, 'steganalysis', 'test_summary.json')
UPLOAD_DIR   = os.path.join(BASE, 'steganalysis', 'uploads')

CLEAN_DIR  = os.path.join(BASE, 'Dataset', 'Clean-1')
STEGO_DIRS = [
    os.path.join(BASE, 'Dataset', 'JMiPOD'),
    os.path.join(BASE, 'Dataset', 'JUNIWARD'),
    os.path.join(BASE, 'Dataset', 'UERD'),
]

os.makedirs(UPLOAD_DIR, exist_ok=True)

#  SRM KERNELS
SRM_KERNELS = [
    np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],
               [-2,8,-12,8,-2],[2,-6,8,-6,2],
               [-1,2,-2,2,-1]], dtype=np.float32) / 12.0,
    np.array([[0,0,0,0,0],[0,-1,2,-1,0],
               [0,2,-4,2,0],[0,-1,2,-1,0],
               [0,0,0,0,0]], dtype=np.float32) / 4.0,
    np.array([[-1,0,1],[0,0,0],
               [1,0,-1]], dtype=np.float32) / 2.0,
    np.array([[1,-2,1],[-2,4,-2],
               [1,-2,1]], dtype=np.float32) / 4.0,
    np.array([[0,1,0],[1,-4,1],
               [0,1,0]], dtype=np.float32),
    np.array([[-1,-1,-1],[-1,8,-1],
               [-1,-1,-1]], dtype=np.float32) / 8.0,
]

def apply_srm_channels(img_bgr):
    channels = []
    for c in range(3):
        ch = img_bgr[:, :, c] / 255.0
        for k in SRM_KERNELS:
            r = cv2.filter2D(ch, -1, k)
            r = r / (np.std(r) + 1e-8)
            channels.append(r.astype(np.float32))
    return np.stack(channels, axis=-1)

# ============================================================
#   LOAD MODEL (once, at startup)
# ============================================================
print('📦 Loading model...')
model = load_model(MODEL_PATH, compile=False, safe_mode=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(f'✅ Model ready — input: {model.input_shape}\n')

# ============================================================
#   SHARED BATCH STATE  (thread-safe via a lock)
# ============================================================
_batch_lock  = threading.Lock()
_batch_state = {
    'running'   : False,
    'done'      : False,
    'processed' : 0,
    'total'     : 0,
    'accuracy'  : None,
    'auc'       : None,
    'f1'        : None,
    'cm'        : None,
    'errors'    : 0,
    'log'       : [],
    'started_at': None,
}

def _update_state(**kwargs):
    with _batch_lock:
        _batch_state.update(kwargs)

def _get_state():
    with _batch_lock:
        return dict(_batch_state)

def _append_log(line):
    with _batch_lock:
        _batch_state['log'].append(line)
        _batch_state['log'] = _batch_state['log'][-50:]

# ============================================================
#   SINGLE IMAGE PREDICT
# ============================================================
def predict_single(image_path, n_augments=4):
    img = cv2.imread(image_path)
    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    variants = [
        img,
        np.fliplr(img),
        np.flipud(img),
        np.fliplr(np.flipud(img)),
    ]

    preds = []
    for v in variants[:n_augments]:
        try:
            ch   = apply_srm_channels(v)
            ch   = np.expand_dims(ch, 0)
            pred = float(model.predict(ch, verbose=0)[0][0])
            preds.append(pred)
        except Exception as e:
            print(f'⚠️  TTA variant error: {e}')

    if not preds:
        return None

    avg   = float(np.mean(preds))
    conf  = avg if avg > 0.5 else 1.0 - avg
    stego = avg > 0.5

    # ── Risk calibrated for ~61% accuracy model ──────────────
    # conf is always 0.50–1.00 (distance from decision boundary).
    # A 61% model rarely exceeds 0.75, so thresholds are scaled
    # down to spread severity across the real score distribution.
    if   conf >= 0.60: risk = 'CRITICAL'   # very confident stego
    elif conf >= 0.57: risk = 'HIGH'        # strong signal
    elif conf >= 0.52: risk = 'MEDIUM'      # moderate signal
    elif conf >= 0.50: risk = 'LOW'         # weak signal
    else:              risk = 'RARE'        # barely above threshold

    return {
        'filename'  : os.path.basename(image_path),
        'score'     : round(avg, 4),
        'confidence': round(conf * 100, 2),
        'is_stego'  : stego,
        'risk'      : risk if stego else 'NONE',
        'timestamp' : datetime.now().isoformat(),
    }

# ============================================================
#  ALERT HELPERS
# ============================================================
def _load_alerts():
    if not os.path.exists(ALERT_LOG):
        return []
    try:
        with open(ALERT_LOG) as f:
            return json.load(f)
    except Exception:
        return []

def _save_alerts(alerts):
    with open(ALERT_LOG, 'w') as f:
        json.dump(alerts, f, indent=2)

def save_alert(result):
    alert = {
        'alert_id'  : os.urandom(4).hex(),
        'timestamp' : result['timestamp'],
        'severity'  : result['risk'],
        'filename'  : result['filename'],
        'confidence': result['confidence'],
        'score'     : result['score'],
        'message'   : f"[{result['risk']}] Steganography in '{result['filename']}'",
    }
    alerts = _load_alerts()
    alerts.insert(0, alert)
    _save_alerts(alerts)

# ============================================================
# WRITE SUMMARY JSON
# ============================================================
def write_summary(preds, true_labels, scores, errors, status):
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    p = np.array(preds)
    l = np.array(true_labels)
    s = np.array(scores)

    acc = float(np.mean(p == l)) * 100

    try:
        auc = float(roc_auc_score(l, s)) if len(set(l.tolist())) > 1 else 0.0
    except Exception:
        auc = 0.0
    try:
        f1 = float(f1_score(l, p))
    except Exception:
        f1 = 0.0
    try:
        cm = confusion_matrix(l, p).tolist()
    except Exception:
        cm = [[0, 0], [0, 0]]

    summary = {
        'accuracy' : round(acc, 2),
        'auc'      : round(auc, 4),
        'f1'       : round(f1, 4),
        'total'    : len(preds),
        'errors'   : errors,
        'cm'       : cm,
        'timestamp': datetime.now().isoformat(),
        'status'   : status,
    }
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    # Mirror into shared state so /api/batch_status responds instantly
    _update_state(
        accuracy=summary['accuracy'],
        auc=summary['auc'],
        f1=summary['f1'],
        cm=cm,
        processed=len(preds),
        errors=errors,
    )

# ============================================================
#   BATCH EVALUATION THREAD
# ============================================================
def run_batch_thread(max_per_class=500):
    try:
        _update_state(running=True, done=False, started_at=datetime.now().isoformat())
        _append_log('🚀 Batch started')

        paths, labels = [], []

        # Collect clean images
        clean_files = [
            f for f in os.listdir(CLEAN_DIR)
            if os.path.isfile(os.path.join(CLEAN_DIR, f))
        ][:max_per_class]
        for f in clean_files:
            paths.append(os.path.join(CLEAN_DIR, f))
            labels.append(0)
        _append_log(f'🟢 Clean: {len(clean_files)} images')

        # Collect stego images
        per_stego = max_per_class // len(STEGO_DIRS)
        for sd in STEGO_DIRS:
            files = [
                f for f in os.listdir(sd)
                if os.path.isfile(os.path.join(sd, f))
            ][:per_stego]
            for f in files:
                paths.append(os.path.join(sd, f))
                labels.append(1)
            _append_log(f'🔴 {os.path.basename(sd)}: {len(files)} images')

        total = len(paths)
        _update_state(total=total)
        _append_log(f'📊 Total: {total} images — starting...')

        preds, scores, errors = [], [], 0

        for idx, path in enumerate(paths):
            result = predict_single(path, n_augments=4)

            if result is None:
                errors += 1
                preds.append(0)
                scores.append(0.0)
            else:
                preds.append(1 if result['is_stego'] else 0)
                scores.append(result['score'])
                # Save ALL results to alert feed — stego with risk, clean as NONE
                alert = {
                    'alert_id'  : os.urandom(4).hex(),
                    'timestamp' : result['timestamp'],
                    'severity'  : result['risk'],   # 'NONE' for clean images
                    'filename'  : result['filename'],
                    'confidence': result['confidence'],
                    'score'     : result['score'],
                    'message'   : (
                        f"[{result['risk']}] Steganography in '{result['filename']}'"
                        if result['is_stego']
                        else f"[NONE] Clean image: '{result['filename']}'"
                    ),
                }
                alerts = _load_alerts()
                alerts.insert(0, alert)
                _save_alerts(alerts)

            # Update progress counter every single image
            _update_state(processed=idx + 1, errors=errors)

            # Every 100 images: compute metrics + write summary + log
            if (idx + 1) % 100 == 0:
                current_acc = np.mean(
                    np.array(preds) == np.array(labels[:len(preds)])
                ) * 100
                msg = (f'Progress {idx+1}/{total} '
                       f'| Acc: {current_acc:.1f}% '
                       f'| Alerts: {sum(preds)} '
                       f'| Errors: {errors}')
                print(f'  {msg}')
                _append_log(msg)

                try:
                    write_summary(
                        preds, labels[:len(preds)],
                        scores, errors, status='running'
                    )
                except Exception as e:
                    print(f'  ⚠️ write_summary error: {e}')

        # Final summary
        write_summary(preds, labels[:len(preds)], scores, errors, status='complete')
        _update_state(running=False, done=True)
        _append_log('✅ Batch complete!')
        print('\n✅ Batch evaluation complete!')

    except Exception as e:
        _update_state(running=False, done=False)
        _append_log(f'❌ Batch error: {e}')
        print(f'❌ Batch thread crashed: {e}')
        raise

# ============================================================
#  FLASK APP
# ============================================================
app = Flask(__name__)

DASHBOARD_HTML = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StegoSentinel</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#050810; --surface:#0d1117; --surface2:#131920;
    --border:#1e2d3d; --accent:#00f5c4; --accent3:#ffd60a;
    --text:#e6edf3; --muted:#6e7681;
    --critical:#ff4d6d; --high:#ff7b00; --medium:#ffd60a;
    --low:#00f5c4; --none:#3fb950;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(0,245,196,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,245,196,.03) 1px,transparent 1px);
    background-size:40px 40px;pointer-events:none;z-index:0}
  .glow{position:fixed;width:600px;height:600px;border-radius:50%;
    background:radial-gradient(circle,rgba(0,245,196,.06) 0%,transparent 70%);
    top:-200px;left:-200px;pointer-events:none;
    animation:orb 20s ease-in-out infinite alternate;z-index:0}
  .glow2{position:fixed;width:400px;height:400px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,77,109,.05) 0%,transparent 70%);
    bottom:-100px;right:-100px;pointer-events:none;z-index:0}
  @keyframes orb{from{transform:translate(0,0)}to{transform:translate(100px,80px)}}

  header{position:relative;z-index:10;padding:20px 40px;display:flex;
    align-items:center;justify-content:space-between;
    border-bottom:1px solid var(--border);background:rgba(5,8,16,.85);backdrop-filter:blur(12px)}
  .logo{display:flex;align-items:center;gap:14px}
  .logo-icon{width:42px;height:42px;background:var(--accent);border-radius:10px;
    display:flex;align-items:center;justify-content:center;font-size:20px;
    box-shadow:0 0 20px rgba(0,245,196,.4);animation:pulse 3s ease-in-out infinite}
  @keyframes pulse{0%,100%{box-shadow:0 0 20px rgba(0,245,196,.4)}50%{box-shadow:0 0 40px rgba(0,245,196,.8)}}
  .logo-text{font-size:22px;font-weight:800;letter-spacing:-.5px}
  .logo-sub{font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;margin-top:2px}
  .header-right{display:flex;align-items:center;gap:16px}
  .pill{display:flex;align-items:center;gap:8px;background:rgba(0,245,196,.08);
    border:1px solid rgba(0,245,196,.2);border-radius:100px;padding:6px 14px;
    font-size:12px;font-family:'Space Mono',monospace;color:var(--accent)}
  .dot{width:7px;height:7px;background:var(--accent);border-radius:50%;animation:blink 1.5s ease-in-out infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
  #clock{font-family:'Space Mono',monospace;font-size:12px;color:var(--muted)}

  #progress-banner{display:none;position:sticky;top:0;z-index:20;
    background:rgba(0,245,196,.06);border-bottom:1px solid rgba(0,245,196,.15);
    padding:10px 40px;font-family:'Space Mono',monospace;font-size:12px;color:var(--accent)}
  #progress-bar-wrap{height:3px;background:var(--border);margin-top:6px;border-radius:100px;overflow:hidden}
  #progress-bar-fill{height:100%;background:linear-gradient(90deg,var(--accent),#00ddb0);width:0%;transition:width .5s ease}

  main{position:relative;z-index:5;padding:32px 40px;max-width:1400px;margin:0 auto}

  .batch-bar{display:flex;align-items:center;gap:12px;margin-bottom:24px;
    background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 24px;
    flex-wrap:wrap}
  .batch-bar h3{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;flex:1;min-width:120px}
  .btn{padding:10px 20px;border-radius:8px;border:none;font-family:'Syne',sans-serif;
    font-size:13px;font-weight:700;cursor:pointer;transition:all .2s;letter-spacing:.3px}
  .btn-primary{background:var(--accent);color:#000}
  .btn-primary:hover:not(:disabled){background:#00ddb0;transform:translateY(-1px)}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed}
  .btn-danger{background:rgba(255,77,109,.15);color:var(--critical);border:1px solid rgba(255,77,109,.3)}
  .btn-danger:hover:not(:disabled){background:rgba(255,77,109,.25)}
  #batch-status-text{font-family:'Space Mono',monospace;font-size:12px;color:var(--muted)}

  .stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
    padding:24px;position:relative;overflow:hidden;transition:transform .2s,border-color .2s}
  .stat-card:hover{transform:translateY(-2px);border-color:var(--accent)}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
  .stat-card.danger::before{background:linear-gradient(90deg,var(--critical),transparent)}
  .stat-card.success::before{background:linear-gradient(90deg,var(--none),transparent)}
  .stat-card.info::before{background:linear-gradient(90deg,var(--accent),transparent)}
  .stat-card.warn::before{background:linear-gradient(90deg,var(--accent3),transparent)}
  .stat-label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;
    color:var(--muted);font-family:'Space Mono',monospace;margin-bottom:10px}
  .stat-value{font-size:36px;font-weight:800;line-height:1;margin-bottom:6px}
  .stat-sub{font-size:12px;color:var(--muted);font-family:'Space Mono',monospace}
  .stat-card.danger .stat-value{color:var(--critical)}
  .stat-card.success .stat-value{color:var(--none)}
  .stat-card.warn .stat-value{color:var(--accent3)}
  .stat-card.info .stat-value{color:var(--accent)}

  .content-grid{display:grid;grid-template-columns:1fr 380px;gap:20px;margin-bottom:20px}
  .panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden}
  .panel-header{padding:18px 24px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between}
  .panel-title{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px}
  .panel-badge{font-size:11px;font-family:'Space Mono',monospace;
    background:rgba(0,245,196,.1);color:var(--accent);
    border:1px solid rgba(0,245,196,.2);padding:3px 10px;border-radius:100px}

  .upload-zone{margin:24px;border:2px dashed var(--border);border-radius:12px;
    padding:32px 20px;text-align:center;cursor:pointer;transition:all .3s;position:relative;overflow:hidden}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);background:rgba(0,245,196,.04)}
  .upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .upload-icon{font-size:36px;margin-bottom:12px}
  .upload-text{font-size:14px;color:var(--muted)}
  .upload-text span{color:var(--accent)}
  .scan-btn{margin:0 24px 24px;width:calc(100% - 48px);padding:14px;
    background:var(--accent);color:#000;border:none;border-radius:10px;
    font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
    cursor:pointer;transition:all .2s;letter-spacing:.5px}
  .scan-btn:hover:not(:disabled){background:#00ddb0;transform:translateY(-1px)}
  .scan-btn:disabled{opacity:.5;cursor:not-allowed}
  .result-box{margin:0 24px 24px;display:none}
  .result-box.show{display:block}
  .result-card{border-radius:12px;padding:20px;border:1px solid}
  .result-card.stego{background:rgba(255,77,109,.08);border-color:rgba(255,77,109,.3)}
  .result-card.clean{background:rgba(63,185,80,.08);border-color:rgba(63,185,80,.3)}
  .result-verdict{font-size:18px;font-weight:800;margin-bottom:12px}
  .result-verdict.stego{color:var(--critical)}
  .result-verdict.clean{color:var(--none)}
  .result-meta{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
  .result-meta-item label{display:block;font-size:10px;color:var(--muted);
    font-family:'Space Mono',monospace;text-transform:uppercase;margin-bottom:3px}
  .result-meta-item span{font-size:14px;font-weight:600;font-family:'Space Mono',monospace}
  .conf-bar-wrap{margin-top:14px}
  .conf-bar-label{display:flex;justify-content:space-between;
    font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;margin-bottom:6px}
  .conf-bar{height:6px;background:var(--border);border-radius:100px;overflow:hidden}
  .conf-bar-fill{height:100%;border-radius:100px;transition:width .8s cubic-bezier(.4,0,.2,1)}
  .conf-bar-fill.stego{background:linear-gradient(90deg,var(--accent3),var(--critical))}
  .conf-bar-fill.clean{background:linear-gradient(90deg,var(--accent),var(--none))}

  .alerts-list{max-height:500px;overflow-y:auto;padding:16px}
  .alert-item{padding:14px 16px;border-radius:10px;border:1px solid var(--border);
    margin-bottom:10px;background:var(--surface2);transition:border-color .2s;
    animation:slidein .4s ease}
  @keyframes slidein{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
  .alert-item:hover{border-color:var(--accent)}
  .alert-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
  .alert-sev{font-size:10px;font-family:'Space Mono',monospace;font-weight:700;
    padding:3px 8px;border-radius:4px;letter-spacing:.5px}
  .sev-CRITICAL{background:rgba(255,77,109,.15);color:var(--critical)}
  .sev-HIGH{background:rgba(255,123,0,.15);color:var(--high)}
  .sev-MEDIUM{background:rgba(255,214,10,.15);color:var(--medium)}
  .sev-LOW{background:rgba(0,245,196,.15);color:var(--low)}
  .sev-RARE{background:rgba(110,118,129,.15);color:#8b949e}
  .sev-NONE{background:rgba(63,185,80,.15);color:var(--none)}
  .alert-time{font-size:10px;color:var(--muted);font-family:'Space Mono',monospace}
  .alert-fname{font-size:13px;font-weight:600;margin-bottom:4px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .alert-conf{font-size:11px;color:var(--muted);font-family:'Space Mono',monospace}
  .no-alerts{text-align:center;padding:40px;color:var(--muted);font-size:13px}
  .no-alerts .icon{font-size:32px;margin-bottom:10px}

  .bottom-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  .metrics-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:24px}
  .metric-item{background:var(--surface2);border:1px solid var(--border);
    border-radius:10px;padding:16px;text-align:center}
  .metric-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;
    color:var(--muted);font-family:'Space Mono',monospace;margin-bottom:8px}
  .metric-value{font-size:26px;font-weight:800;color:var(--accent)}
  .cm-wrap{padding:0 24px 24px}
  .cm-title{font-size:11px;text-transform:uppercase;letter-spacing:1px;
    color:var(--muted);font-family:'Space Mono',monospace;margin-bottom:14px}
  .cm-grid{display:grid;grid-template-columns:auto 1fr 1fr;
    gap:6px;font-family:'Space Mono',monospace;font-size:12px}
  .cm-header{color:var(--muted);text-align:center;padding:6px}
  .cm-row-label{color:var(--muted);display:flex;align-items:center;padding-right:10px}
  .cm-cell{border-radius:8px;padding:16px 8px;text-align:center;font-weight:700;font-size:18px}
  .cm-tp,.cm-tn{background:rgba(63,185,80,.15);color:var(--none)}
  .cm-fp,.cm-fn{background:rgba(255,77,109,.15);color:var(--critical)}

  #batch-log{font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);
    padding:0 24px 24px;max-height:180px;overflow-y:auto;line-height:1.9}
  #batch-log div{border-bottom:1px solid rgba(255,255,255,.03);padding:2px 0}

  .loading-overlay{display:none;position:fixed;inset:0;background:rgba(5,8,16,.85);
    backdrop-filter:blur(6px);z-index:100;align-items:center;justify-content:center;
    flex-direction:column;gap:20px}
  .loading-overlay.show{display:flex}
  .spinner{width:50px;height:50px;border:3px solid var(--border);
    border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .loading-text{font-size:14px;color:var(--muted);font-family:'Space Mono',monospace;
    animation:fade-pulse 1.5s ease-in-out infinite}
  @keyframes fade-pulse{0%,100%{opacity:.5}50%{opacity:1}}

  .toast-container{position:fixed;bottom:30px;right:30px;z-index:200;display:flex;flex-direction:column;gap:10px}
  .toast{padding:14px 20px;border-radius:10px;font-size:13px;font-family:'Space Mono',monospace;
    animation:toast-in .3s ease;max-width:340px}
  .toast.success{background:rgba(63,185,80,.15);border:1px solid rgba(63,185,80,.3);color:var(--none)}
  .toast.error{background:rgba(255,77,109,.15);border:1px solid rgba(255,77,109,.3);color:var(--critical)}
  .toast.warn{background:rgba(255,214,10,.15);border:1px solid rgba(255,214,10,.3);color:var(--medium)}
  @keyframes toast-in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

  @media(max-width:1100px){
    .stats-grid{grid-template-columns:repeat(2,1fr)}
    .content-grid,.bottom-grid{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<div class="glow"></div><div class="glow2"></div>

<div class="loading-overlay" id="loading">
  <div class="spinner"></div>
  <div class="loading-text">Analyzing image...</div>
</div>
<div class="toast-container" id="toasts"></div>

<header>
  <div class="logo">
    <div class="logo-icon">&#128300;</div>
    <div>
      <div class="logo-text">StegoSentinel</div>
      <div class="logo-sub">STEGANALYSIS DETECTION SYSTEM</div>
    </div>
  </div>
  <div class="header-right">
    <div class="pill"><div class="dot"></div>MODEL ACTIVE</div>
    <div id="clock"></div>
  </div>
</header>

<div id="progress-banner">
  <span id="progress-text">&#9889; Batch running...</span>
  <div id="progress-bar-wrap"><div id="progress-bar-fill"></div></div>
</div>

<main>

  <div class="batch-bar">
    <h3>&#128202; Batch Evaluation</h3>
    <span id="batch-status-text">Idle — click Run Batch to start</span>
    <button class="btn btn-primary" id="btn-run" onclick="startBatch()">&#9654; Run Batch</button>
    <button class="btn btn-danger"  id="btn-clear" onclick="clearAlerts()">&#128465; Clear Alerts</button>
  </div>

  <div class="stats-grid">
    <div class="stat-card danger">
      <div class="stat-label">Total Alerts</div>
      <div class="stat-value" id="stat-alerts">—</div>
      <div class="stat-sub">steganography detected</div>
    </div>
    <div class="stat-card success">
      <div class="stat-label">Model Accuracy</div>
      <div class="stat-value" id="stat-accuracy">—</div>
      <div class="stat-sub">on test dataset</div>
    </div>
    <div class="stat-card info">
      <div class="stat-label">ROC-AUC Score</div>
      <div class="stat-value" id="stat-auc">—</div>
      <div class="stat-sub">discrimination power</div>
    </div>
    <div class="stat-card warn">
      <div class="stat-label">F1 Score</div>
      <div class="stat-value" id="stat-f1">—</div>
      <div class="stat-sub">precision x recall</div>
    </div>
  </div>

  <div class="content-grid">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">&#128444; Image Scanner</div>
        <div class="panel-badge">TTA x 8 variants</div>
      </div>
      <div class="upload-zone" id="drop-zone">
        <input type="file" id="file-input" accept="image/*">
        <div class="upload-icon">&#128193;</div>
        <div class="upload-text">Drop an image here or <span>browse files</span></div>
        <div class="upload-text" style="margin-top:8px;font-size:12px" id="file-name">PNG, JPG, BMP, PGM supported</div>
      </div>
      <button class="scan-btn" id="scan-btn" onclick="scanImage()">
        &#128269; SCAN FOR STEGANOGRAPHY
      </button>
      <div class="result-box" id="result-box">
        <div class="result-card" id="result-card">
          <div class="result-verdict" id="result-verdict"></div>
          <div class="conf-bar-wrap">
            <div class="conf-bar-label"><span>Confidence</span><span id="conf-pct"></span></div>
            <div class="conf-bar"><div class="conf-bar-fill" id="conf-fill" style="width:0%"></div></div>
          </div>
          <div class="result-meta">
            <div class="result-meta-item"><label>Risk Level</label><span id="res-risk"></span></div>
            <div class="result-meta-item"><label>Raw Score</label><span id="res-score"></span></div>
            <div class="result-meta-item"><label>File Name</label><span id="res-file" style="font-size:11px"></span></div>
            <div class="result-meta-item"><label>Action</label><span id="res-action" style="font-size:11px"></span></div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">&#128680; Alert Feed</div>
        <div class="panel-badge" id="alert-count-badge">0 alerts</div>
      </div>
      <div class="alerts-list" id="alerts-list">
        <div class="no-alerts"><div class="icon">&#128737;</div>No alerts yet.<br>Run a batch or scan an image.</div>
      </div>
    </div>
  </div>

  <div class="bottom-grid">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">&#128202; Model Performance</div>
        <div class="panel-badge" id="perf-badge">Awaiting batch</div>
      </div>
      <div class="metrics-grid">
        <div class="metric-item"><div class="metric-label">Accuracy</div><div class="metric-value" id="m-accuracy">—</div></div>
        <div class="metric-item"><div class="metric-label">AUC Score</div><div class="metric-value" id="m-auc">—</div></div>
        <div class="metric-item"><div class="metric-label">F1 Score</div><div class="metric-value" id="m-f1">—</div></div>
        <div class="metric-item"><div class="metric-label">Tested</div><div class="metric-value" id="m-total">—</div></div>
      </div>
      <div class="cm-wrap">
        <div class="cm-title">Confusion Matrix</div>
        <div class="cm-grid">
          <div></div>
          <div class="cm-header">Pred Clean</div>
          <div class="cm-header">Pred Stego</div>
          <div class="cm-row-label">True Clean</div>
          <div class="cm-cell cm-tn" id="cm-tn">—</div>
          <div class="cm-cell cm-fp" id="cm-fp">—</div>
          <div class="cm-row-label">True Stego</div>
          <div class="cm-cell cm-fn" id="cm-fn">—</div>
          <div class="cm-cell cm-tp" id="cm-tp">—</div>
        </div>
      </div>
      <div class="cm-title" style="padding:16px 24px 8px">Batch Log</div>
      <div id="batch-log">No batch run yet.</div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">&#8505; System Info</div>
        <div class="panel-badge">Alaska2 Dataset</div>
      </div>
      <div style="padding:24px;display:flex;flex-direction:column;gap:14px">
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:16px">
          <div style="font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Architecture</div>
          <div style="font-size:14px;font-weight:600">Residual CNN + TLU Activation</div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px">4 blocks · 18-channel SRM RGB input</div>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:16px">
          <div style="font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Training Dataset</div>
          <div style="font-size:14px;font-weight:600">Alaska2 — 40,000 images</div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px">JMiPOD · JUNIWARD · UERD</div>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:16px">
          <div style="font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Hardware</div>
          <div style="font-size:14px;font-weight:600">NVIDIA RTX 4050 · i7-13700H</div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px">WSL2 · TensorFlow 2.15 · CUDA 12.3</div>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:16px">
          <div style="font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Inference</div>
          <div style="font-size:14px;font-weight:600">Test-Time Augmentation (TTA)</div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px">8 geometric variants averaged</div>
        </div>
      </div>
    </div>
  </div>

</main>

<script>
// ── Clock ─────────────────────────────────────────────────────
setInterval(() => {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-US', {hour12: false});
}, 1000);

// ── Start batch ───────────────────────────────────────────────
async function startBatch() {
  document.getElementById('btn-run').disabled = true;
  try {
    const r = await fetch('/api/start_batch', {method: 'POST'});
    const d = await r.json();
    if (d.error) {
      showToast(d.error, 'warn');
      document.getElementById('btn-run').disabled = false;
    } else {
      showToast('Batch started! Results will appear live.', 'success');
    }
  } catch(e) {
    showToast('Could not start batch: ' + e.message, 'error');
    document.getElementById('btn-run').disabled = false;
  }
}

async function clearAlerts() {
  await fetch('/api/clear_alerts', {method: 'POST'});
  showToast('Alerts cleared', 'success');
  loadAlerts();
}

// ── Poll /api/batch_status every 3 s ─────────────────────────
async function pollBatchStatus() {
  try {
    const r = await fetch('/api/batch_status');
    const d = await r.json();

    const banner = document.getElementById('progress-banner');
    const fill   = document.getElementById('progress-bar-fill');
    const ptext  = document.getElementById('progress-text');
    const btnRun = document.getElementById('btn-run');
    const stText = document.getElementById('batch-status-text');

    if (d.running) {
      banner.style.display = 'block';
      btnRun.disabled      = true;
      const pct = d.total > 0 ? Math.round(d.processed / d.total * 100) : 0;
      fill.style.width  = pct + '%';
      ptext.textContent = `⚡ Batch running — ${d.processed} / ${d.total} images (${pct}%)`;
      stText.textContent = `Running… ${d.processed}/${d.total}`;
    } else {
      banner.style.display = 'none';
      btnRun.disabled      = false;
      stText.textContent   = d.done
        ? `Complete — ${d.processed} images evaluated`
        : 'Idle — click Run Batch to start';
    }

    // Update metric cards directly from in-memory state
    if (d.accuracy !== null) {
      document.getElementById('stat-accuracy').textContent = d.accuracy + '%';
      document.getElementById('stat-auc').textContent      = d.auc;
      document.getElementById('stat-f1').textContent       = d.f1;
      document.getElementById('m-accuracy').textContent    = d.accuracy + '%';
      document.getElementById('m-auc').textContent         = d.auc;
      document.getElementById('m-f1').textContent          = d.f1;
      document.getElementById('m-total').textContent       = d.processed;
      document.getElementById('perf-badge').textContent    = d.running ? 'LIVE' : 'Complete';
      if (d.cm) {
        document.getElementById('cm-tn').textContent = d.cm[0][0];
        document.getElementById('cm-fp').textContent = d.cm[0][1];
        document.getElementById('cm-fn').textContent = d.cm[1][0];
        document.getElementById('cm-tp').textContent = d.cm[1][1];
      }
    }

    // Batch log
    if (d.log && d.log.length) {
      document.getElementById('batch-log').innerHTML =
        d.log.slice().reverse().map(l => `<div>${l}</div>`).join('');
    }
  } catch(e) {}
}

// ── Poll /api/alerts every 5 s ────────────────────────────────
async function loadAlerts() {
  try {
    const r      = await fetch('/api/alerts');
    const alerts = await r.json();
    document.getElementById('stat-alerts').textContent       = alerts.length;
    document.getElementById('alert-count-badge').textContent = alerts.length + ' alerts';
    renderAlerts(alerts);
  } catch(e) {}
}

function renderAlerts(alerts) {
  const el = document.getElementById('alerts-list');
  if (!alerts.length) {
    el.innerHTML = `<div class="no-alerts"><div class="icon">&#128737;</div>No alerts yet.<br>Run a batch or scan an image.</div>`;
    return;
  }
  el.innerHTML = alerts.slice(0, 30).map(a => `
    <div class="alert-item">
      <div class="alert-top">
        <span class="alert-sev sev-${a.severity}">${a.severity}</span>
        <span class="alert-time">${new Date(a.timestamp).toLocaleTimeString()}</span>
      </div>
      <div class="alert-fname">${a.filename}</div>
      <div class="alert-conf">Confidence: ${a.confidence}%</div>
    </div>`).join('');
}

// ── Image scanner ─────────────────────────────────────────────
let selectedFile = null;

document.getElementById('file-input').addEventListener('change', e => {
  selectedFile = e.target.files[0];
  if (selectedFile) document.getElementById('file-name').textContent = selectedFile.name;
});

const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', ()  => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag-over');
  selectedFile = e.dataTransfer.files[0];
  if (selectedFile) document.getElementById('file-name').textContent = selectedFile.name;
});

async function scanImage() {
  if (!selectedFile) { showToast('Please select an image first', 'warn'); return; }
  document.getElementById('loading').classList.add('show');
  document.getElementById('scan-btn').disabled = true;
  const form = new FormData();
  form.append('image', selectedFile);
  try {
    const r   = await fetch('/api/predict', {method: 'POST', body: form});
    const res = await r.json();
    if (res.error) { showToast('Error: ' + res.error, 'error'); return; }
    showResult(res);
    loadAlerts();
    showToast(
      res.is_stego
        ? `&#128680; Steganography detected (${res.confidence}%)`
        : `&#9989; Clean image (${res.confidence}% confidence)`,
      res.is_stego ? 'error' : 'success'
    );
  } catch(e) {
    showToast('Scan failed: ' + e.message, 'error');
  } finally {
    document.getElementById('loading').classList.remove('show');
    document.getElementById('scan-btn').disabled = false;
  }
}

function showResult(res) {
  const cls = res.is_stego ? 'stego' : 'clean';
  document.getElementById('result-card').className    = 'result-card ' + cls;
  document.getElementById('result-verdict').className = 'result-verdict ' + cls;
  document.getElementById('result-verdict').innerHTML = res.is_stego
    ? '&#9888; STEGANOGRAPHY DETECTED' : '&#9989; CLEAN IMAGE';
  document.getElementById('conf-pct').textContent   = res.confidence + '%';
  document.getElementById('res-risk').textContent   = res.risk || 'NONE';
  document.getElementById('res-score').textContent  = res.score;
  document.getElementById('res-file').textContent   = res.filename;
  document.getElementById('res-action').textContent = res.action || '';
  const fill = document.getElementById('conf-fill');
  fill.className   = 'conf-bar-fill ' + cls;
  fill.style.width = '0%';
  setTimeout(() => { fill.style.width = res.confidence + '%'; }, 50);
  document.getElementById('result-box').classList.add('show');
}

function showToast(msg, type = 'success') {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.innerHTML = msg;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0'; t.style.transition = 'opacity .3s';
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

// ── Start polling immediately ─────────────────────────────────
pollBatchStatus();
loadAlerts();
setInterval(pollBatchStatus, 3000);
setInterval(loadAlerts,      5000);
</script>
</body>
</html>
'''

# ============================================================
#   ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/start_batch', methods=['POST'])
def start_batch():
    state = _get_state()
    if state['running']:
        return jsonify({'error': 'Batch already running'})
    _update_state(
        running=False, done=False, processed=0, total=0,
        accuracy=None, auc=None, f1=None, cm=None,
        errors=0, log=[], started_at=None,
    )
    _save_alerts([])   # clear old alerts for fresh run
    t = threading.Thread(target=run_batch_thread, args=(500,), daemon=True)
    t.start()
    return jsonify({'status': 'started'})


@app.route('/api/batch_status')
def batch_status():
    return jsonify(_get_state())


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'No file selected'})

    filename = secure_filename(file.filename)
    path     = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Cannot read image file'})
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    variants = [
        img, np.fliplr(img), np.flipud(img), np.fliplr(np.flipud(img)),
        np.rot90(img, 1), np.rot90(img, 2), np.rot90(img, 3),
        np.clip(img + np.random.normal(0, 0.5, img.shape).astype(np.float32), 0, 255),
    ]

    preds = []
    for v in variants:
        try:
            ch = apply_srm_channels(v)
            ch = np.expand_dims(ch, 0)
            preds.append(float(model.predict(ch, verbose=0)[0][0]))
        except Exception as e:
            print(f'⚠️  TTA error: {e}')

    if not preds:
        return jsonify({'error': 'Prediction failed'})

    avg   = float(np.mean(preds))
    conf  = avg if avg > 0.5 else 1.0 - avg
    stego = avg > 0.5

    # ── Risk calibrated for ~61% accuracy model ──────────────
    if   conf >= 0.75: risk = 'CRITICAL'
    elif conf >= 0.65: risk = 'HIGH'
    elif conf >= 0.57: risk = 'MEDIUM'
    elif conf >= 0.53: risk = 'LOW'
    else:              risk = 'RARE'

    result = {
        'filename'  : filename,
        'score'     : round(avg, 4),
        'confidence': round(conf * 100, 2),
        'is_stego'  : stego,
        'risk'      : risk if stego else 'NONE',
        'result'    : 'STEGO DETECTED' if stego else 'CLEAN',
        'action'    : 'Forensic analysis recommended' if stego else 'No action required',
        'timestamp' : datetime.now().isoformat(),
    }

    if stego:
        save_alert(result)

    return jsonify(result)


@app.route('/api/alerts')
def get_alerts():
    return jsonify(_load_alerts())


@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    _save_alerts([])
    return jsonify({'status': 'cleared'})


@app.route('/api/summary')
def get_summary():
    if not os.path.exists(SUMMARY_PATH):
        return jsonify({})
    try:
        with open(SUMMARY_PATH) as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({})


# ============================================================
#   RUN — single app.run() call
# ============================================================
if __name__ == '__main__':
    print('=' * 52)
    print('  🌐  StegoSentinel Dashboard')
    print('  📍  http://localhost:5000')
    print('  ▶️   Open the browser and click "Run Batch"')
    print('=' * 52 + '\n')
    app.run(host='0.0.0.0', port=5000, debug=False)