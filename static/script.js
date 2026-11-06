const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const resultText = document.getElementById("resultText");
const predictButton = document.getElementById("predictButton");
const previewBox = document.getElementById("previewBox");
const uploadBox = document.querySelector(".upload-box");

if (uploadBox && imageInput) {
  uploadBox.addEventListener("click", () => imageInput.click());
}

// Show image preview if elements exist
if (imageInput && previewImage && previewBox) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewBox.style.display = "block";
      };
      reader.readAsDataURL(file);
      if (resultText) resultText.textContent = "";
    }
  });
}

// Handle Predict Button if asynchronous API is used
if (predictButton && imageInput) {
  predictButton.addEventListener("click", () => {
    const file = imageInput.files[0];
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    if (resultText) resultText.textContent = "🔍 Analyzing...";
    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        if (!resultText) return;
        if (data.disease) {
          resultText.textContent = "🩺 Result: " + data.disease.replace(/___/g, " ");
        } else {
          resultText.textContent = "⚠️ Error: " + data.error;
        }
      })
      .catch((err) => {
        console.error(err);
        if (resultText) resultText.textContent = "❌ Something went wrong. Try again!";
      });
  });
}

// Modal functions
function openHistory() {
  const modal = document.getElementById('historyModal');
  if (modal) modal.style.display = 'flex';
}

function closeHistory() {
  const modal = document.getElementById('historyModal');
  if (modal) modal.style.display = 'none';
}
