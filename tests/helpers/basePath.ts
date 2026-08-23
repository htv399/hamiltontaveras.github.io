const configuredBase = process.env.BASE_PATH?.replace(/^\/+|\/+$/g, "") ?? "";

export function withBase(route: string): string {
  const normalizedRoute = route.startsWith("/") ? route : `/${route}`;
  return configuredBase ? `/${configuredBase}${normalizedRoute}` : normalizedRoute;
}
