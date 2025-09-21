import kociemba

def solve_cube(cube_state):
    """
    cube_state: 54-char string representing cube
    Example: "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    """
    try:
        solution = kociemba.solve(cube_state)
        return solution.split()
    except Exception as e:
        return [f"Error: {e}"]
