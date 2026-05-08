# InterviewIQ

InterviewIQ is an AI mock interview web app with voice, video, scorecards, n8n report delivery, and an Express backend.

## Local Run

Backend:

```powershell
cd backend
npm install
npm start
```

Frontend:

Open `index.html` in a browser, or serve the repo root with any static server.

## Netlify Frontend

Deploy the repo root to Netlify.

- Build command: leave empty
- Publish directory: `.`

The frontend points to:

```text
https://interviewiq-backend-navya29.onrender.com
```

You can override this in the browser console:

```js
localStorage.setItem('INTERVIEWIQ_API_BASE', 'https://your-backend-url.onrender.com')
location.reload()
```

## Render Backend

Create a Render Blueprint from this repo using `render.yaml`, or create a Web Service manually:

- Root directory: `backend`
- Build command: `npm install`
- Start command: `npm start`
- Plan: Free

Required environment variables:

- `AI_PROVIDER=groq`
- `AI_MODEL=llama-3.1-8b-instant`
- `GROQ_API_KEY`
- `N8N_REPORT_WEBHOOK_URL`

The backend uses SQLite on the service filesystem. On a free ephemeral host, saved history may reset after redeploys or restarts.
