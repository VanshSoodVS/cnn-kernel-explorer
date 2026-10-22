'use strict';
(() => {
  const script = document.currentScript;
  const snapshot = script.dataset.snapshot;
  const title = document.getElementById('loading-title');
  const note = document.getElementById('loading-note');
  const box = document.getElementById('loading');
  const retry = document.getElementById('retry-load');
  let started = false;
  async function load() {
    if (started) return;
    started = true;
    retry.hidden = true;
    box.classList.remove('error');
    try {
      if (typeof DecompressionStream === 'undefined') {
        throw new Error('This browser cannot open the compressed snapshot. Please open this page in a current version of Safari, Chrome, Firefox or Edge.');
      }
      title.textContent = 'Loading this snapshot';
      note.textContent = 'Downloading saved weights and matches. On a mobile connection this can take a moment.';
      const response = await fetch('data/' + snapshot + '.json.gz');
      if (!response.ok) throw new Error('The snapshot download failed (HTTP ' + response.status + '). Please try again.');
      const bytes = new Uint8Array(await response.arrayBuffer());
      title.textContent = 'Preparing the kernel maps';
      note.textContent = 'Unpacking the saved results in your browser. No training is running.';
      // A host may transparently decompress a .gz file. Detect the gzip header
      // instead of attempting a second decompression in that case.
      const compressed = bytes[0] === 0x1f && bytes[1] === 0x8b;
      const json = compressed
        ? await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()
        : new TextDecoder().decode(bytes);
      window.KERNEL_DATA = JSON.parse(json);
      if (window.KERNEL_DATA.snapshot !== snapshot) throw new Error('The downloaded snapshot does not match this page. Please reload it.');
      document.getElementById('explorer-main').hidden = false;
      // Give the loading message a frame before decoding large typed arrays.
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      await new Promise((resolve, reject) => {
        const viewer = document.createElement('script');
        viewer.src = 'viewer.js?v=mobile-1';
        viewer.onload = resolve;
        viewer.onerror = () => reject(new Error('The explorer controls could not load. Please reload this page.'));
        document.body.append(viewer);
      });
      if (!window.KERNEL_VIEWER_READY) throw new Error('The explorer could not prepare this snapshot. Please reload the page.');
      box.hidden = true;
    } catch (error) {
      document.getElementById('explorer-main').hidden = true;
      title.textContent = 'This snapshot could not be opened';
      note.textContent = error.message || 'Please check your connection and reload this page.';
      box.classList.add('error');
      retry.hidden = false;
      retry.onclick = () => window.location.reload();
    }
  }
  load();
})();
