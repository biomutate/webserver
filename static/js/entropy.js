// Function to generate the file upload section based on the selected number of files
function generateFileUploadSection() {
    const numFiles = parseInt(document.getElementById('num-files-select').value);
    const fileUploadSection = document.getElementById('file-upload-section');
    fileUploadSection.innerHTML = '';

    for (let i = 0; i < numFiles; i++) {
        const fileDiv = document.createElement('div');
        fileDiv.innerHTML = `
            <div class="file-info">
                <div>
                    <label for="file_${i}">Upload file ${i + 1}</label>
                    <input type="file" name="file_${i}">
                </div>
                <div>
                    <label for="month_${i}">Month for file ${i + 1}</label>
                    <input type="text" name="month_${i}">
                </div>
                <div>
                    <label for="year_${i}">Year for file ${i + 1}</label>
                    <input type="text" name="year_${i}">
                </div>
            </div>
         `;
         fileUploadSection.appendChild(fileDiv);
    }
}

// Add an event listener to the num_files select element
document.getElementById('num-files-select').addEventListener('change', generateFileUploadSection);

// Generate the initial file upload section on page load
generateFileUploadSection();
