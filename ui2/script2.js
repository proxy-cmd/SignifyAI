const startBtn = document.getElementById("startBtn");
const typewriterText = document.getElementById("typewriterText");

const words = [
  "Signify AI",
  "Computer Vision",
  "Smart Detection",
  "Visual Intelligence"
];

let wordIndex = 0;
let charIndex = 0;
let isDeleting = false;

function typeEffect() {
  const currentWord = words[wordIndex];
  const visibleText = currentWord.slice(0, charIndex);

  typewriterText.textContent = visibleText;

  let typingSpeed = isDeleting ? 70 : 130;

  if (!isDeleting && charIndex < currentWord.length) {
    charIndex++;
  } else if (!isDeleting && charIndex === currentWord.length) {
    typingSpeed = 1400;
    isDeleting = true;
  } else if (isDeleting && charIndex > 0) {
    charIndex--;
  } else {
    isDeleting = false;
    wordIndex = (wordIndex + 1) % words.length;
    typingSpeed = 250;
  }

  setTimeout(typeEffect, typingSpeed);
}

typeEffect();

startBtn.addEventListener("click", () => {
  startBtn.innerText = "Starting Vision Engine...";

  setTimeout(() => {
    window.location.href = "main.html";
  }, 600);
});
