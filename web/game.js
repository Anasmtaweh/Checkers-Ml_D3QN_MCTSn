// --- GLOBAL STATE ---
let selectedCell = null;
let currentLegalMoves = [];
let currentPlayer = 1;
let isGameOver = false;

// Auto-Play State
let autoPlayInterval = null;
let autoPlayEnabled = false; // The safety flag
let aiMoveInFlight = false;  // Prevents double-firing

async function startGame() {
    const p1 = document.getElementById('p1-select').value;
    const p2 = document.getElementById('p2-select').value;
    
    // Toggle AI controls visibility
    const isAIvsAI = (p1 !== 'human' || p2 !== 'human');
    const aiControls = document.getElementById('ai-controls');
    if (aiControls) aiControls.style.display = isAIvsAI ? 'block' : 'none';
    
    // Reset Auto-Play state on new game
    stopAutoPlay();

    try {
        const response = await fetch('/start_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ p1, p2 })
        });
        const state = await response.json();
        renderBoard(state);
    } catch (e) {
        console.error("Start failed:", e);
    }
}

// fromAuto = true if called by the Interval, false if called by "AI Move" button or post-human move
async function triggerAIMove(fromAuto = false) {
    if (isGameOver) return;
    
    // Prevent network spam if one is already loading
    if (aiMoveInFlight) return;

    aiMoveInFlight = true;

    try {
        const response = await fetch('/get_move', { method: 'POST' });
        const state = await response.json();

        // If server says "Wait" or "Error", just exit
        if (state.error) {
            // console.log("Server waiting...", state.error);
            return;
        }

        // --- CRITICAL FIX: PHANTOM MOVE PREVENTION ---
        // If this request came from the Auto-Play loop, but the user clicked "Stop"
        // while the request was travelling over the network, we MUST DROP this packet.
        // Otherwise, the board updates 0.5s after you clicked Stop.
        if (fromAuto && !autoPlayEnabled) {
            console.log("Packet dropped (Auto-Play was stopped).");
            return; 
        }

        renderBoard(state);
        
    } catch (e) {
        console.error("AI move failed:", e);
    } finally {
        aiMoveInFlight = false;
    }
}

function toggleAutoPlay() {
    if (autoPlayInterval) {
        stopAutoPlay();
    } else {
        autoPlayEnabled = true;
        const btn = document.getElementById('auto-btn');
        if (btn) btn.innerText = "⏸️ Stop Auto";
        
        // Trigger immediately, then interval
        triggerAIMove(true);
        // 500ms delay between moves for visual clarity
        autoPlayInterval = setInterval(() => triggerAIMove(true), 500); 
    }
}

function stopAutoPlay() {
    autoPlayEnabled = false; // Kill flag immediately
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
    }
    const btn = document.getElementById('auto-btn');
    if (btn) btn.innerText = "▶️ Auto-Play";
}

function renderBoard(state) {
    const board = document.getElementById('board');
    if (!board) return;

    board.innerHTML = '';
    currentLegalMoves = state.legal_moves || [];
    currentPlayer = state.current_player;
    isGameOver = state.game_over;
    selectedCell = null; // Deselect on update

    // Status Text
    const statusObj = document.getElementById('status-text');
    if (statusObj) {
        statusObj.innerText = state.game_over ? "GAME OVER" : 
            (state.current_player === 1 ? "🔴 Red's Turn" : "⚫ Black's Turn");
    }
    
    // Winner Text
    const winnerObj = document.getElementById('winner-display');
    if (winnerObj) {
        if (state.winner !== 0) {
            winnerObj.innerText = state.winner === 1 ? "🏆 RED WINS!" : "🏆 BLACK WINS!";
            stopAutoPlay(); // Safety stop
        } else {
            winnerObj.innerText = "";
        }
    }

    // Draw Grid
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement('div');
            cell.className = `cell ${(r + c) % 2 === 0 ? 'light' : 'dark'}`;
            cell.dataset.r = r;
            cell.dataset.c = c;
            cell.onclick = () => onCellClick(r, c);

            // --- NEW: LAST MOVE HIGHLIGHT ---
            if (state.last_move) {
                const lm = state.last_move;
                // Check Start
                if (lm.start[0] === r && lm.start[1] === c) {
                    cell.classList.add('last-move');
                }
                // Check End
                if (lm.end[0] === r && lm.end[1] === c) {
                    cell.classList.add('last-move');
                }
            }
            // --------------------------------

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
    if (isGameOver) return;

    // 1. Try to Execute Move
    // Check if the clicked cell is a valid landing spot for the previously selected cell
    const move = currentLegalMoves.find(m => 
        selectedCell && 
        m[0][0] === selectedCell.r && m[0][1] === selectedCell.c && 
        m[1][0] === r && m[1][1] === c
    );

    if (move) {
        try {
            const response = await fetch('/human_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ move: move })
            });
            const newState = await response.json();
            
            if (newState.error) {
                console.log("Move rejected:", newState.error);
                return;
            }

            renderBoard(newState);
            
            // If playing Human vs AI, trigger AI response
            // We pass false because this is a "manual" chain reaction, not the auto-loop
            if (!newState.game_over) {
                 setTimeout(() => triggerAIMove(false), 500); 
            }

        } catch (e) {
            console.error("Human move failed:", e);
        }
        return;
    }

    // 2. Select a Piece
    // Highlight if this cell starts a legal move
    const isStartOfMove = currentLegalMoves.some(m => m[0][0] === r && m[0][1] === c);
    
    if (isStartOfMove) {
        selectedCell = { r, c };
        
        // Redraw highlights
        const cells = document.querySelectorAll('.cell');
        cells.forEach(c => c.classList.remove('highlight'));
        
        // Highlight current piece
        const currentIdx = r * 8 + c;
        if (cells[currentIdx]) cells[currentIdx].classList.add('highlight');

        // Highlight targets
        currentLegalMoves.forEach(m => {
            if (m[0][0] === r && m[0][1] === c) {
                const targetR = m[1][0];
                const targetC = m[1][1];
                const idx = targetR * 8 + targetC;
                if (cells[idx]) cells[idx].classList.add('highlight');
            }
        });
    } else {
        // Deselect if clicking empty space or invalid piece
        selectedCell = null;
        document.querySelectorAll('.cell').forEach(c => c.classList.remove('highlight'));
    }
}