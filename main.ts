// A push-optimal Sokoban solver: one Node 22 program, standard library only.
//
// The board is a hallway with the player plus one to four small rooms, each holding
// three or four boxes and as many goals. A box that leaves its room through the door
// can never come back, so every room is an independent subproblem and the level's
// optimum is the sum of the rooms' optima. Each room is solved with a breadth-first
// search over pushes: a state is the boxes' squares together with the player's
// reachable region (named by its lowest square), so the first solved state the search
// meets is push-optimal for that room. Two deadlocks are pruned, both provably safe:
// a box on a square from which no push can ever reach a goal, and a box caught in a
// frozen 2x2 block of walls and boxes with a box off its goal. The search is capped
// by a count of states, never by the clock, and every tie is broken by fixed square
// and direction order, so two runs write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The most states one room search keeps before it gives up: work, never time. */
const MAX_STATES = 250_000;

/** The four steps, in the order ties are broken: up, down, left, right. */
const DIRS = [
  { dx: 0, dy: -1, step: 'u' },
  { dx: 0, dy: 1, step: 'd' },
  { dx: -1, dy: 0, step: 'l' },
  { dx: 1, dy: 0, step: 'r' },
];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

/** Read XSB text into a board: `#` wall, `.` goal, `$` box, `*` box on goal, `@`/`+` player. */
function parse(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever the text shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

/**
 * The squares a lone box can reach a goal from: pull back from the goals. A box on
 * square y can be pushed to x = y + d exactly when the player can stand on y - d,
 * so the live set grows backwards from every goal through pairs of open squares.
 */
function liveSquares(board: Board): Uint8Array {
  const { w, wall, goal } = board;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (let i = 0; i < wall.length; i++) {
    if (goal[i]) {
      live[i] = 1;
      queue.push(i);
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const { dx, dy } of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood the squares the player reaches from `from` with `occupied` in the way,
 * marking them with `stamp` in `seen`; returns the lowest square of the region.
 * The lowest square names the region: two states with the same boxes and the same
 * lowest reachable square give the player the same pushes.
 */
function regionLow(
  board: Board,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = board;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * Whether the box just pushed to `t` is stuck in a frozen 2x2 block: four squares
 * of walls and boxes holding a box off its goal. A block like that never changes,
 * so the state is dead.
 */
function frozen(board: Board, occupied: Uint8Array, t: number): boolean {
  const { w, wall, goal } = board;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower-case steps. */
function walkTo(board: Board, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = board;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const { dx, dy, step } of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The player's square after playing `moves` (case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The push-optimal solution of one board: { pushes, moves } with pushes in upper
 * case inside the move string, or null when the search gives up or the board has
 * no solution. States are kept in the order found with their parent, so the
 * solution is rebuilt by walking back from the first solved state.
 */
function solvePushes(board: Board, maxStates = MAX_STATES): { pushes: number; moves: string } | null {
  const { w, wall, goal } = board;
  const size = wall.length;
  const live = liveSquares(board);
  if (board.boxes.some((b) => !live[b])) return null;
  const done = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const stateBoxes: number[][] = [board.boxes];
  const statePlayer: number[] = [board.player];
  const parent: number[] = [-1];
  const keyOf = (boxes: number[], low: number): string => `${low}|${boxes.join(',')}`;

  for (const b of board.boxes) occupied[b] = 1;
  const firstLow = regionLow(board, occupied, board.player, seen, ++stamp, stack);
  for (const b of board.boxes) occupied[b] = 0;
  const index = new Map<string, number>([[keyOf(board.boxes, firstLow), 0]]);

  let found = done(board.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < stateBoxes.length; head++) {
    const boxes = stateBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    regionLow(board, occupied, statePlayer[head]!, seen, ++stamp, stack);
    const here = stamp; // the stamp naming this state's reachable region
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const { dx, dy } of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (wall[t] || occupied[t] || !live[t]) continue;
        if (seen[b - d] !== here) continue; // the player cannot reach the push square
        occupied[b] = 0;
        occupied[t] = 1;
        let next: number[] | null = null;
        let key = '';
        if (!frozen(board, occupied, t)) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, x) => a - x);
          key = keyOf(next, regionLow(board, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (next === null || index.has(key)) continue;
        if (stateBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, stateBoxes.length);
        stateBoxes.push(next);
        statePlayer.push(b); // after the push the player stands on the box's old square
        parent.push(head);
        if (done(next)) {
          found = stateBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = board.player;
  let boxes = board.boxes;
  for (const s of path) {
    const nextBoxes = stateBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(({ dx, dy }) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(board, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir.step.toUpperCase();
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * Split a level into rooms: cut the dead floor squares of the player's row (the
 * hallway — a box there reaches no goal) out of the floor, and each connected part
 * left that holds boxes is a room. Null when a box starts in the hallway, a goal
 * falls outside every part, or some part's boxes and goals do not match.
 */
function roomsOf(board: Board): Room[] | null {
  const { w, wall, goal, boxes, player } = board;
  const live = liveSquares(board);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const { dx, dy } of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (let g = 0; g < wall.length; g++) {
    if (goal[g]) {
      if (part[g] < 0) return null;
      rooms[part[g]!]!.goals.push(g);
    }
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Solve a level room by room: each room's push-optimal solution in turn, the other
 * rooms' boxes standing as walls (on their goals once their room is done), the
 * player walking on from where the last room left it. Falls back to solving the
 * whole level at once when it is no set of rooms.
 */
function solveLevel(board: Board): string | null {
  const rooms = roomsOf(board);
  if (rooms === null) {
    const whole = solvePushes(board);
    return whole === null ? null : whole.moves;
  }
  let player = board.player;
  let moves = '';
  for (const room of rooms) {
    const wall = board.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    // Only this room's goals count for the sub-board: other rooms' goals are walls.
    const subGoal = new Uint8Array(board.wall.length);
    for (const g of room.goals) subGoal[g] = 1;
    const sub: Board = {
      w: board.w,
      h: board.h,
      wall,
      goal: subGoal,
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushes(sub);
    if (solved === null) return null;
    moves += solved.moves;
    player = endOf(board.w, player, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(IN).sort()) {
  const moves = solveLevel(parse(readFileSync(join(IN, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), `${moves ?? ''}\n`);
}
