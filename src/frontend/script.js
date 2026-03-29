const API_URL = 'http://localhost:8000';

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const result = document.getElementById('result');
const error = document.getElementById('error');

let selectedFile = null;

// Выбор файла
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Обработка файла
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Пожалуйста, выберите изображение (JPG, PNG)');
        return;
    }
    
    selectedFile = file;
    hideError();
    hideResult();
    
    // Показ превью
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        preview.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Блокируем кнопку
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Анализ...';
    hideError();
    hideResult();
    
    try {
        // Отправка на апи
        const formData = new FormData();
        formData.append('file_bytes', selectedFile);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Ошибка при анализе');
        }   
        
        const data = await response.json();
        showResult(data);
        
    } catch (err) {
        showError('Ошибка при анализе');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Проанализировать';
    }
});

function showResult(data) {
    document.getElementById('diseaseRu').textContent = data.disease_rus;
    document.getElementById('diseaseEn').textContent = data.disease;
    document.getElementById('confidence').textContent = 
        (data.confidence * 100).toFixed(1) + '%';
    result.classList.remove('hidden');

    console.log(data);
}

function showError(message) {
    error.textContent = message;
    error.classList.remove('hidden');
}

function hideError() {
    error.classList.add('hidden');
}

function hideResult() {
    result.classList.add('hidden');
}