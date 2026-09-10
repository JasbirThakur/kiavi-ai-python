# ==============================================================================
# Kiavi IQ - Frontend Application Dockerfile
# Service: Nginx Production Web Server & Reverse Proxy (Port 80 -> Host 3000)
# ==============================================================================
FROM nginx:alpine

LABEL maintainer="Kiavi AI Team"
LABEL service="kiavi-frontend-ui"
LABEL description="Web Dashboard, Embeddable Chat Widget, and Live Chat Interface"

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy custom reverse-proxy and routing configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy all frontend files (HTML, JS, CSS, widget assets)
COPY public /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
