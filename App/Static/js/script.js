 /*const uploadContainer = document.querySelector('.upload-container');
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, preventDefaults);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        uploadContainer.addEventListener('drop', handleDrop);
        imageInput.addEventListener('change', handleSelect);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const file = dt.files[0];
            handleFile(file);
        }

        function handleSelect(e) {
            const file = e.target.files[0];
            handleFile(file);
        }

        function handleFile(file) {
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);

                uploadImage(file);
            }
        }

        async function uploadImage(file) {
            const formData = new FormData();
            formData.append('image', file);

            try {
                result.textContent = 'Processing...';
                const response = await fetch('/count_leaves', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    result.textContent = data.message;
                    result.classList.remove('error');
                } else {
                    result.textContent = data.error;
                    result.classList.add('error');
                }
            } catch (error) {
                result.textContent = 'Error processing image';
                result.classList.add('error');
            }
        }*/

        module.exports = {
        theme: {
            extend: {
            animation: {
                marquee: 'marquee 25s linear infinite',
                marquee2: 'marquee2 25s linear infinite',
            },
            keyframes: {
                marquee: {
                '0%': { transform: 'translateX(0%)' },
                '100%': { transform: 'translateX(-100%)' }
                },
                marquee2: {
                '0%': { transform: 'translateX(100%)' },
                '100%': { transform: 'translateX(0%)' }
                },
            },
            },
        },
        }
              