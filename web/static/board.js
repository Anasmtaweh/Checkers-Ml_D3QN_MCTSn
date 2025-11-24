const canvas = document.getElementById("boardCanvas");
const ctx = canvas.getContext("2d");
const tileSize = 60; // 480 / 8

let board = Array.from({ length: 8 }, () => Array(8).fill(0));
let score = {};

function drawBoard() {
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            ctx.fillStyle = (r + c) % 2 === 0 ? "#444" : "#222";
            ctx.fillRect(c * tileSize, r * tileSize, tileSize, tileSize);

            const val = board[r][c];
            if (val === 1) drawPiece(c, r, "red");
            if (val === -1) drawPiece(c, r, "white");
            if (val === 2) drawKing(c, r, "red");
            if (val === -2) drawKing(c, r, "white");
        }
    }
}

function drawPiece(x, y, color) {
    ctx.beginPath();
    ctx.arc(x * tileSize + tileSize / 2, y * tileSize + tileSize / 2, 22, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawKing(x, y, color) {
    ctx.beginPath();
    ctx.arc(x * tileSize + tileSize / 2, y * tileSize + tileSize / 2, 22, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "gold";
    ctx.lineWidth = 4;
    ctx.stroke();
}

async function startGame(mode) {
    const res = await fetch("/start_game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode })
    });
    const data = await res.json();
    score = data.score || {};
    await updateState();
}

async function updateState() {
    const res = await fetch("/get_state");
    const data = await res.json();
    board = data.board || board;
    score = data.score || score;
    updateScoreboard();
    drawBoard();
}

async function stepGame() {
    const res = await fetch("/play_step", { method: "POST" });
    const data = await res.json();
    if (data.board) {
        board = data.board;
    }
    if (data.score) {
        score = data.score;
    }
    updateScoreboard();
    drawBoard();
}

// Initial paint
drawBoard();

function updateScoreboard() {
    const redAgent = document.getElementById("red-agent");
    const whiteAgent = document.getElementById("white-agent");
    const redCounts = document.getElementById("red-counts");
    const whiteCounts = document.getElementById("white-counts");
    const turnEl = document.getElementById("turn");

    if (!score || Object.keys(score).length === 0) return;

    const agents = score.agents || {};
    const pieces = score.pieces || {};
    const current = score.current_player;

    if (redAgent) redAgent.textContent = `Red (Player 1) – ${agents.red || "?"} Agent`;
    if (whiteAgent) whiteAgent.textContent = `White (Player 2) – ${agents.white || "?"} Agent`;

    if (redCounts) {
        const rp = pieces.red ? pieces.red.pieces : 0;
        const rk = pieces.red ? pieces.red.kings : 0;
        redCounts.textContent = `Red: ${rp} pieces, ${rk} kings`;
    }
    if (whiteCounts) {
        const wp = pieces.white ? pieces.white.pieces : 0;
        const wk = pieces.white ? pieces.white.kings : 0;
        whiteCounts.textContent = `White: ${wp} pieces, ${wk} kings`;
    }
    if (turnEl) {
        const turnStr = current === 1 ? "Red" : current === -1 ? "White" : "?";
        turnEl.textContent = `Turn: ${turnStr}`;
    }
}
