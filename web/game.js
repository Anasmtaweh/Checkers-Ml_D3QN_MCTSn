let selectedCell = null;
let currentLegalMoves = [];
let currentPlayer = 1;
let isGameOver = false;
let autoPlayInterval = null;

async function startGame() {
    const p1 = document.getElementById('p1-select').value;
    const p2 = document.getElementById('p2-select').value;
    
    // Toggle AI controls
    const isAIvsAI = (p1 !== 'human' || p2 !== 'human');
    document.getElementById('ai-controls').style.display = isAIvsAI ? 'block' : 'none';
    stopAutoPlay();

    const response = await fetch('/start_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ p1, p2 })
    });
    const state = await response.json();
    renderBoard(state);
}

async function triggerAIMove() {
    if (isGameOver) return;
    
    const response = await fetch('/get_move', { method: 'POST' });
    const state = await response.json();
    
    if (state.error) {
        console.log("Waiting for human...");
        return;
    }
    renderBoard(state);
}

function toggleAutoPlay() {
    if (autoPlayInterval) {
        stopAutoPlay();
    } else {
        document.getElementById('auto-btn').innerText = "⏸️ Stop Auto";
        autoPlayInterval = setInterval(triggerAIMove, 500); // 0.5s per move
    }
}

function stopAutoPlay() {
    clearInterval(autoPlayInterval);
    autoPlayInterval = null;
    document.getElementById('auto-btn').innerText = "▶️ Auto-Play";
}

function renderBoard(state) {
    const board = document.getElementById('board');
    board.innerHTML = '';
    currentLegalMoves = state.legal_moves;
    currentPlayer = state.current_player;
    isGameOver = state.game_over;
    selectedCell = null;

    document.getElementById('status-text').innerText = 
        state.game_over ? "GAME OVER" : 
        (state.current_player === 1 ? "🔴 Red's Turn" : "⚫ Black's Turn");
    
    if (state.winner !== 0) {
        document.getElementById('winner-display').innerText = 
            state.winner === 1 ? "🏆 RED WINS!" : "🏆 BLACK WINS!";
        stopAutoPlay();
    } else {
        document.getElementById('winner-display').innerText = "";
    }

    // Draw Grid
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement('div');
            cell.className = `cell ${(r + c) % 2 === 0 ? 'light' : 'dark'}`;
            cell.dataset.r = r;
            cell.dataset.c = c;
            cell.onclick = () => onCellClick(r, c);

            const pieceVal = state.board[r][c];
            if (pieceVal !== 0) {
                const piece = document.createElement('div');
                piece.className = `piece ${pieceVal > 0 ? 'red' : 'black'} ${Math.abs(pieceVal) === 2 ? 'king' : ''}`;
                cell.appendChild(piece);
            }
            board.appendChild(cell);
        }
    }
}

async function onCellClick(r, c) {
    // Only handle click if it's Human turn
    // (Backend will reject if we try to move for AI, but nice to block here too)
    
    // 1. If selecting a piece
    const board = document.querySelectorAll('.cell');
    
    // Clear highlights
    board.forEach(c => c.classList.remove('highlight'));

    // Check if we clicked a valid move destination
    const move = currentLegalMoves.find(m => 
        selectedCell && 
        m[0][0] === selectedCell.r && m[0][1] === selectedCell.c && 
        m[1][0] === r && m[1][1] === c
    );

    if (move) {
        // EXECUTE MOVE
        const response = await fetch('/human_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: [[selectedCell.r, selectedCell.c], [r, c]] })
        });
        const newState = await response.json();
        renderBoard(newState);
        
        // If playing against AI, trigger AI response after short delay
        setTimeout(triggerAIMove, 500); 
        return;
    }

    // 2. Select a new piece
    // Check if this cell contains current player's piece
    // (This part requires client to know board state, simplified by checking moves)
    const isStartOfMove = currentLegalMoves.some(m => m[0][0] === r && m[0][1] === c);
    
    if (isStartOfMove) {
        selectedCell = { r, c };
        // Highlight possible targets
        currentLegalMoves.forEach(m => {
            if (m[0][0] === r && m[0][1] === c) {
                const targetR = m[1][0];
                const targetC = m[1][1];
                const idx = targetR * 8 + targetC;
                board[idx].classList.add('highlight');
            }
        });
    }
}