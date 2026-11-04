'use strict';
// the oauth consent page, after "allow": copies the callback address for apps that ask for it
document.getElementById('copy-callback').addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('callback').textContent.trim())
        .then(() => { document.getElementById('copied').hidden = false; })
        .catch(() => { window.getSelection().selectAllChildren(document.getElementById('callback')); });
});
