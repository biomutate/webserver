document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var form = event.target;
    var formData = new FormData(form);

    fetch('/preprocessing', {
                method: 'POST',
                body: formData
            })
            .then(function(response) {
                if (response.ok) {
                    document.getElementById('uploadStatus').textContent = 'Upload successful.';
                } else {
                    document.getElementById('uploadStatus').textContent = 'Upload failed. Please try again.';
                }
                document.getElementById('uploadStatus').style.display = 'block';
            })
            .catch(function(error) {
                console.error('Error:', error);
                document.getElementById('uploadStatus').textContent = 'Upload failed. Please try again.';
                document.getElementById('uploadStatus').style.display = 'block';
            });
        });