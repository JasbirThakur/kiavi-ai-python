(function() {
    const scriptTag = document.currentScript;
    const publicKey = scriptTag.getAttribute('data-key');
    
    if (!publicKey) {
        console.error('Kiavi IQ: Missing data-key attribute on script tag.');
        return;
    }

    // Determine host URL automatically based on script source
    const hostUrl = new URL(scriptTag.src).origin;
    
    // Create Iframe
    const iframe = document.createElement('iframe');
    iframe.src = `${hostUrl}/widget/${publicKey}`;
    iframe.style.position = 'fixed';
    iframe.style.bottom = '20px';
    // Default to right, will be updated by iframe postMessage
    iframe.style.right = '20px';
    iframe.style.width = '80px';
    iframe.style.height = '80px';
    iframe.style.border = 'none';
    iframe.style.zIndex = '999999999';
    iframe.style.overflow = 'hidden';
    iframe.style.transition = 'width 0.3s ease, height 0.3s ease';
    iframe.style.colorScheme = 'light';
    
    document.body.appendChild(iframe);

    // Listen for messages from the iframe (to resize or move it)
    window.addEventListener('message', function(event) {
        if (event.origin !== hostUrl) return;
        
        const data = event.data;
        if (data.type === 'resize') {
            iframe.style.width = data.width;
            iframe.style.height = data.height;
        }
        if (data.type === 'position') {
            if (data.pos === 'left') {
                iframe.style.right = 'auto';
                iframe.style.left = '20px';
            } else {
                iframe.style.left = 'auto';
                iframe.style.right = '20px';
            }
        }
    });
})();