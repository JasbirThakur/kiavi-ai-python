FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy custom reverse-proxy and routing configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy all frontend files (HTML, JS, assets)
COPY public /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
