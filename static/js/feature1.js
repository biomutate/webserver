// Submit form and handle response
const form = document.getElementById('upload-form');
const errorText = document.getElementById('error');
const resultsSection = document.getElementById('results');
const downloadLink = document.getElementById('download-link');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    errorText.style.display = 'none';
    resultsSection.style.display = 'none';

    const formData = new FormData(form);
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then((response) => {
        if (response.ok) {
            response.blob().then((blob) => {
                const url = window.URL.createObjectURL(blob);
                downloadLink.href = url;
                downloadLink.style.display = 'inline';
                resultsSection.style.display = 'block';
            });
        } else {
            errorText.style.display = 'block';
        }
    })
    .catch((error) => {
        errorText.style.display = 'block';
    });
});

document.getElementById('upload-form').addEventListener('submit', (e) => {
    e.preventDefault();

    const errorText = document.getElementById('error');
    const resultsSection = document.getElementById('results');
    const downloadLink = document.getElementById('download-link');

    errorText.style.display = 'none';
    resultsSection.style.display = 'none';

    const formData = new FormData(e.target);
    fetch('/calculate_feature1', {
      method: 'POST',
      body: formData
    })
    .then((response) => {
      if (response.ok) {
        response.blob().then((blob) => {
          const url = window.URL.createObjectURL(blob);
          downloadLink.href = url;
          downloadLink.style.display = 'inline';
          resultsSection.style.display = 'block';
        });
      } else {
        errorText.style.display = 'block';
      }
    })
    .catch((error) => {
      errorText.style.display = 'block';
    });
});