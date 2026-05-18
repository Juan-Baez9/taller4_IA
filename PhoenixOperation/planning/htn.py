from __future__ import annotations

from planning.utils import Queue
from collections import deque
from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.
 
    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.
 
    Returns a list of primitive Action objects, or [] if no plan found.
    """
    initial_plan = list(hlas)
    frontier = Queue()
    frontier.push(initial_plan)
    visited_plans = set()

    while not frontier.isEmpty():
        plan = frontier.pop()

        plan_key = tuple(id(step) for step in plan)
        if plan_key in visited_plans:
            continue
        visited_plans.add(plan_key)

        if is_plan_primitive(plan):
            state = problem.getStartState()
            valid = True
            for action in plan:
                if is_applicable(state, action):
                    state = apply_action(state, action)
                else:
                    valid = False
                    break
            if valid and problem.isGoalState(state):
                return plan
            continue

        first_idx = next(i for i, step in enumerate(plan) if not is_primitive(step))
        hla = plan[first_idx]

        for refinement in hla.refinements:
            new_plan = plan[:first_idx] + list(refinement) + plan[first_idx + 1:]
            frontier.push(new_plan)

    return []
        
def _bfs_path(start, goal, adjacency: dict) -> list | None:
    if start == goal:
        return [start]
    frontier = deque([[start]])
    visited = {start}
    while frontier:
        path = frontier.popleft()
        current = path[-1]
        for neighbor in adjacency.get(current, []):
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(path + [neighbor])
    return None
 
 
def _path_to_moves(robot, path: list) -> list:
    return [_make_move(robot, path[i], path[i + 1]) for i in range(len(path) - 1)]
 
 
# ---------------------------------------------------------------------------
# Helpers: acciones primitivas con fluentes como tuplas
# ---------------------------------------------------------------------------
 
 
def _make_move(robot, frm, to) -> Action:
    return Action(
        name=f"Move({robot},{frm},{to})",
        precond_pos=[("At", robot, frm), ("Adjacent", frm, to), ("Free", to)],
        precond_neg=[],
        add_list=[("At", robot, to), ("Free", frm)],
        del_list=[("At", robot, frm), ("Free", to)],
    )
 
 
def _make_pickup(robot, obj, loc) -> Action:
    return Action(
        name=f"PickUp({robot},{obj},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("At", obj, loc),
            ("HandsFree", robot),
            ("Pickable", obj),
        ],
        precond_neg=[],
        add_list=[("Holding", robot, obj)],
        del_list=[("At", obj, loc), ("HandsFree", robot)],
    )
 
 
def _make_putdown(robot, obj, loc) -> Action:
    return Action(
        name=f"PutDown({robot},{obj},{loc})",
        precond_pos=[("At", robot, loc), ("Holding", robot, obj)],
        precond_neg=[],
        add_list=[("At", obj, loc), ("HandsFree", robot)],
        del_list=[("Holding", robot, obj)],
    )
 
 
def _make_rescue(robot, patient, loc) -> Action:
    return Action(
        name=f"Rescue({robot},{patient},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("At", patient, loc),
            ("MedicalPost", loc),
            ("SuppliesReady", loc),
        ],
        precond_neg=[],
        add_list=[("Rescued", patient)],
        del_list=[("At", patient, loc)],
    )
 
 
def _make_setup_supplies(robot, supplies, loc) -> Action:
    return Action(
        name=f"SetupSupplies({robot},{supplies},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("MedicalPost", loc),
            ("Holding", robot, supplies),
        ],
        precond_neg=[],
        add_list=[("SuppliesReady", loc), ("HandsFree", robot), ("At", supplies, loc)],
        del_list=[("Holding", robot, supplies)],
    )


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###
    state: frozenset = problem.getStartState()
    objects: dict = problem.objects

    robots = objects.get("robots", [])
    cells = objects.get("cells", [])
    supplies_list = objects.get("supplies", [])
    patients = objects.get("patients", [])

    adjacency: dict = {cell: [] for cell in cells}
    for f in state:
        if isinstance(f, tuple) and len(f) == 3 and f[0] == "Adjacent":
            if f[1] in adjacency:
                adjacency[f[1]].append(f[2])

    medical_posts = []
    for f in state:
        if isinstance(f, tuple) and len(f) == 2 and f[0] == "MedicalPost":
            medical_posts.append(f[1])

    robot = robots[0] if robots else None
    medical_post = medical_posts[0] if medical_posts else None

    if robot is None or medical_post is None:
        return []

    def get_location(entity):
        for f in state:
            if isinstance(f, tuple) and len(f) == 3 and f[0] == "At" and f[1] == entity:
                return f[2]
        return None

    def make_navigate(frm, to) -> HLA:
        path = _bfs_path(frm, to, adjacency)
        if path is None or len(path) < 2:
            return HLA(f"Navigate({frm},{to})", refinements=[])
        moves = _path_to_moves(robot, path)
        return HLA(f"Navigate({frm},{to})", refinements=[moves])

    full_mission_hlas = []

    for s, p in zip(supplies_list, patients):
        s_loc = get_location(s)
        p_loc = get_location(p)
        r_loc = get_location(robot)

        if s_loc is None or p_loc is None or r_loc is None:
            continue

        # --- PrepareSupplies ---
        prep_steps = []
        if r_loc != s_loc:
            prep_steps.append(make_navigate(r_loc, s_loc))
        prep_steps.append(_make_pickup(robot, s, s_loc))
        if s_loc != medical_post:
            prep_steps.append(make_navigate(s_loc, medical_post))
        prep_steps.append(_make_setup_supplies(robot, s, medical_post))

        prepare_hla = HLA(
            name=f"PrepareSupplies({s},{medical_post})",
            refinements=[prep_steps],
        )

        # --- ExtractPatient ---
        extract_steps = []
        if medical_post != p_loc:
            extract_steps.append(make_navigate(medical_post, p_loc))
        extract_steps.append(_make_pickup(robot, p, p_loc))
        if p_loc != medical_post:
            extract_steps.append(make_navigate(p_loc, medical_post))
        extract_steps.append(_make_putdown(robot, p, medical_post))

        extract_hla = HLA(
            name=f"ExtractPatient({p},{medical_post})",
            refinements=[extract_steps],
        )

        # --- FullRescueMission ---
        rescue_action = _make_rescue(robot, p, medical_post)
        full_mission = HLA(
            name=f"FullRescueMission({s},{p},{medical_post})",
            refinements=[[prepare_hla, extract_hla, rescue_action]],
        )
        full_mission_hlas.append(full_mission)

    return full_mission_hlas