export async function apiRequest<T = any>(url: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const isIdempotent = !options.method || ['GET', 'HEAD'].includes(options.method.toUpperCase());
  const maxRetries = isIdempotent ? 4 : 0;
  const retryDelays = [400, 800, 1200, 1600];

  let response: Response | undefined;
  let lastError: any = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      response = await fetch(url, {
        ...options,
        headers,
      });
      if (response && [502, 503, 504].includes(response.status) && attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, retryDelays[attempt] || 1000));
        continue;
      }
      break;
    } catch (err) {
      lastError = err;
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, retryDelays[attempt] || 1000));
        continue;
      }
      throw err;
    }
  }

  if (!response) {
    throw lastError || new Error(`Network request failed for ${url}`);
  }

  if (!response.ok) {
    let errorMsg = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const text = await response.text();
      if (text && text.trim()) {
        try {
          const errJson = JSON.parse(text);
          if (errJson.detail) errorMsg = errJson.detail;
          else if (errJson.message) errorMsg = errJson.message;
          else if (errJson.error) errorMsg = errJson.error;
          else errorMsg = text.trim();
        } catch (_) {
          errorMsg = text.trim();
        }
      }
    } catch (_) {}
    throw new Error(errorMsg);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text() as unknown as T;
}
