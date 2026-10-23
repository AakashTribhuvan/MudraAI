const TUNNEL_HOST_PATTERN = /^[a-z0-9-]+\.trycloudflare\.com$/i;

export class TunnelState {
  constructor(state) {
    this.state = state;
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/target") {
      const target = await this.state.storage.get("target");
      return Response.json({ target: target ?? null });
    }
    if (request.method === "PUT" && url.pathname === "/target") {
      const payload = await request.json();
      await this.state.storage.put("target", payload.target);
      return Response.json({ updated: true });
    }
    return new Response("Not found", { status: 404 });
  }
}

function constantTimeEqual(left, right) {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

async function currentTunnel(env) {
  const state = env.TUNNEL_STATE.getByName("current");
  const response = await state.fetch("https://tunnel-state/target");
  const { target } = await response.json();
  return target;
}

export default {
  async fetch(request, env) {
    const requestUrl = new URL(request.url);

    if (requestUrl.pathname === "/__update") {
      if (request.method !== "POST") {
        return new Response("Method not allowed", {
          status: 405,
          headers: { Allow: "POST" },
        });
      }
      const expectedToken = env.UPDATE_TOKEN;
      const suppliedToken = request.headers.get("Authorization") ?? "";
      if (
        !expectedToken ||
        !constantTimeEqual(suppliedToken, `Bearer ${expectedToken}`)
      ) {
        return new Response("Unauthorized", { status: 401 });
      }

      let payload;
      try {
        payload = await request.json();
      } catch {
        return new Response("Expected a JSON body", { status: 400 });
      }
      let targetUrl;
      try {
        targetUrl = new URL(payload?.target);
      } catch {
        return new Response("Invalid tunnel URL", { status: 400 });
      }
      if (
        targetUrl.protocol !== "https:" ||
        !TUNNEL_HOST_PATTERN.test(targetUrl.hostname) ||
        targetUrl.pathname !== "/" ||
        targetUrl.search ||
        targetUrl.hash
      ) {
        return new Response("Invalid Quick Tunnel URL", { status: 400 });
      }

      const state = env.TUNNEL_STATE.getByName("current");
      return state.fetch(
        new Request("https://tunnel-state/target", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target: targetUrl.origin }),
        }),
      );
    }

    const target = await currentTunnel(env);
    if (!target) {
      return new Response("SignLine is offline. Start the judges launcher on the laptop first.", {
        status: 503,
        headers: { "Cache-Control": "no-store" },
      });
    }

    // Keep the browser on this stable hostname for OAuth callbacks and cookies.
    // The local Quick Tunnel hostname changes on every launch, so it is only an upstream.
    const destination = new URL(requestUrl.pathname + requestUrl.search, target);
    const upstream = new Request(destination, request);
    if (request.headers.get("Upgrade")?.toLowerCase() === "websocket") {
      return fetch(upstream);
    }
    const response = await fetch(upstream, { redirect: "manual" });
    const responseHeaders = new Headers(response.headers);
    responseHeaders.set("Cache-Control", "no-store");
    return new Response([204, 205, 304].includes(response.status) ? null : response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  },
};
