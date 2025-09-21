# solver.py
import kociemba

def solve_kociemba(cube_54_str):
    """
    Input: 54-char cube string where each char is one of 'U','R','F','D','L','B'
           order: faces in URFDLB, each face 9 chars top-left → bottom-right
    Output: list of moves (e.g. ['R','U','R\'','U\''])
    """
    try:
        sol = kociemba.solve(cube_54_str)  # returns string like "R U R' U'"
        moves = sol.strip().split()
        return moves
    except Exception as e:
        raise RuntimeError(f"Solver error: {e}")
