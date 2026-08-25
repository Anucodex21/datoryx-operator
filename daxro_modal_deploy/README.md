# DAXRO fine-tuned - Modal GPU deployment

Yeh folder DAXRO fine-tuned model ko **Modal** pe ek GPU-backed, hamesha-live
API ke roop mein deploy karta hai. Main DATORYX backend (Render pe, CPU-only)
isko HTTP se call karega `DAXRO_REMOTE_URL` env var ke through.

## Steps

### 1. Modal account banao aur CLI install karo

```
pip install modal
modal setup
```

Browser khulega, Modal account se login/signup karo (free tier credits milte hain).

### 2. Deploy karo

Isी folder mein (`daxro_modal_deploy/`) terminal khol ke:

```
cd daxro_modal_deploy
modal deploy app.py
```

Pehli deploy mein base model (Qwen2.5-3B, ~2GB) download hoga ek persistent
volume mein — thoda time lagega (5-10 min). Uske baad deploys/calls fast honge.

### 3. URL copy karo

Deploy complete hone par terminal mein ek URL print hoga, kuch aisa:

```
https://<your-workspace>--daxro-inference-daxro-chat-completions.modal.run
```

### 4. Render mein wire karo

Render dashboard > apni service > Environment mein:

```
DAXRO_REMOTE_URL=https://<your-workspace>--daxro-inference-daxro-chat-completions.modal.run
```

Save karte hi Render service restart hogi, aur DAXRO fine-tuned model ab
production mein live hai — DATORYX backend isko automatically HTTP se call
karega jab bhi `provider="daxro-remote"` use hoga ya jab koi cloud API key
set na ho.

## Testing (deploy se pehle ya baad mein)

```
curl -X POST "https://<your-modal-url>" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a Python function to reverse a string."}]}'
```

Real, working Python code wapas aana chahiye.

## Cost note

Modal `gpu="T4"` use karta hai aur `scaledown_window=300` set hai — matlab
5 minute idle rehne ke baad GPU spin down ho jata hai (paisa bachane ke liye),
aur agli request par phir se load hota hai (thoda cold-start delay, ~10-20s).
Agar tumhe zero cold-start chahiye (company demo ke waqt), `min_containers=1`
add kar sakte ho `@app.cls(...)` mein — lekin isse GPU hamesha chalta rahega
aur zyada cost aayega.

## Security note

Abhi endpoint publicly open hai (koi auth check nahi). Agar chaho to Modal
ke docs dekh ke ek shared secret token add kar sakte ho aur usko
`DAXRO_REMOTE_TOKEN` env var mein daal sakte ho — `DaxroRemoteProvider`
(backend mein) already `Authorization: Bearer <token>` header bhejta hai
agar `DAXRO_REMOTE_TOKEN` set ho.
