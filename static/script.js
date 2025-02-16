const canvas = document.getElementById("matrix");
const ctx = canvas.getContext("2d");

// Set canvas to full screen
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();

// Characters for the matrix effect
const characters = "ABCDEFGHIJKLMNOPQRSTUVXYZABCDEFGHIJKLMNOPQRSTUVXYZ0123456789";
const charactersArray = characters.split("");

const fontSize = 14;
let columns = canvas.width / fontSize;

// Create an array of drops (one per column)
let drops = [];
function initDrops() {
    drops = [];
    for (let i = 0; i < columns; i++) {
        drops[i] = Math.floor(Math.random() * canvas.height / fontSize);
    }
}
initDrops();

// Draw the matrix effect
function draw() {
    // Black background with slight transparency
    ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set text color and font
    ctx.fillStyle = "#0F0"; // Neon green color
    ctx.font = fontSize + "px monospace";

    // Loop through drops
    for (let i = 0; i < drops.length; i++) {
        // Pick a random character
        const text = charactersArray[Math.floor(Math.random() * charactersArray.length)];

        // Draw the character
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        // Reset drop if it reaches the bottom
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
            drops[i] = 0;
        }

        // Move the drop down
        drops[i]++;
    }
}

// Render the matrix effect
setInterval(draw, 50);

// Resize canvas on window resize
window.addEventListener("resize", () => {
    resizeCanvas();
    columns = canvas.width / fontSize;
    initDrops();
});