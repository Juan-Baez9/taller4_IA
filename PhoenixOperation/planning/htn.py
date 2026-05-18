from __future__ import annotations

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
    # El plan inicial es la lista de HLAs de alto nivel (una FullRescueMission por paciente,
    # o simplemente los hlas que recibimos como punto de entrada)
    initial_plan = list(hlas)
 
    # Cola BFS: cada elemento es un plan (lista de HLA/Action)
    frontier = Queue()
    frontier.push(initial_plan)
 
    visited_plans = set()
 
    while not frontier.isEmpty():
        plan = frontier.pop()
 
        # Convertir a clave hashable para evitar ciclos
        plan_key = tuple(id(step) for step in plan)
        if plan_key in visited_plans:
            continue
        visited_plans.add(plan_key)
 
        # Caso base: plan completamente primitivo
        if is_plan_primitive(plan):
            # Verificar que el plan alcanza el objetivo desde el estado inicial
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
 
        # Encontrar el primer paso no primitivo
        first_hla_index = next(
            i for i, step in enumerate(plan) if not is_primitive(step)
        )
        hla = plan[first_hla_index]
 
        # Expandir cada refinamiento posible
        for refinement in hla.refinements:
            # Construir nuevo plan reemplazando la HLA por su refinamiento
            new_plan = plan[:first_hla_index] + list(refinement) + plan[first_hla_index + 1:]
            frontier.push(new_plan)
 
    # No se encontró plan
    return []
 
 
# ---------------------------------------------------------------------------
# Helpers para construir acciones primitivas
# ---------------------------------------------------------------------------
 
 
def _make_move(robot: str, frm: str, to: str) -> Action:
    """Crea una acción Move primitiva instanciada."""
    return Action(
        name=f"Move({robot},{frm},{to})",
        precond_pos=frozenset({f"At({robot},{frm})", f"Adjacent({frm},{to})", f"Free({to})"}),
        precond_neg=frozenset(),
        effect_add=frozenset({f"At({robot},{to})"}),
        effect_del=frozenset({f"At({robot},{frm})"}),
    )
 
 
def _make_pickup(robot: str, obj: str, loc: str) -> Action:
    return Action(
        name=f"PickUp({robot},{obj},{loc})",
        precond_pos=frozenset({
            f"At({robot},{loc})",
            f"At({obj},{loc})",
            f"HandsFree({robot})",
            f"Pickable({obj})",
        }),
        precond_neg=frozenset(),
        effect_add=frozenset({f"Holding({robot},{obj})"}),
        effect_del=frozenset({f"At({obj},{loc})", f"HandsFree({robot})"}),
    )
 
 
def _make_putdown(robot: str, obj: str, loc: str) -> Action:
    return Action(
        name=f"PutDown({robot},{obj},{loc})",
        precond_pos=frozenset({f"At({robot},{loc})", f"Holding({robot},{obj})"}),
        precond_neg=frozenset(),
        effect_add=frozenset({f"At({obj},{loc})", f"HandsFree({robot})"}),
        effect_del=frozenset({f"Holding({robot},{obj})"}),
    )
 
 
def _make_rescue(robot: str, patient: str, loc: str) -> Action:
    return Action(
        name=f"Rescue({robot},{patient},{loc})",
        precond_pos=frozenset({
            f"At({robot},{loc})",
            f"At({patient},{loc})",
            f"MedicalPost({loc})",
            f"SuppliesReady({loc})",
        }),
        precond_neg=frozenset(),
        effect_add=frozenset({f"Rescued({patient})"}),
        effect_del=frozenset({f"At({patient},{loc})"}),
    )
 
 
def _make_setup_supplies(robot: str, supplies: str, loc: str) -> Action:
    return Action(
        name=f"SetupSupplies({robot},{supplies},{loc})",
        precond_pos=frozenset({
            f"At({robot},{loc})",
            f"At({supplies},{loc})",
            f"MedicalPost({loc})",
            f"Holding({robot},{supplies})",
        }),
        precond_neg=frozenset(),
        effect_add=frozenset({f"SuppliesReady({loc})", f"HandsFree({robot})"}),
        effect_del=frozenset({f"Holding({robot},{supplies})"}),
    )
    ### End of your code ###


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
    state = problem.getStartState()
    fluents = state.fluents  # frozenset de strings
 

    robots, cells, supplies_list, patients, medical_posts = [], [], [], [], []
    adjacencies: dict[str, list[str]] = {}  # celda -> vecinos adyacentes
 
    for f in fluents:
        if f.startswith("At("):
            inner = f[3:-1]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                entity, loc = parts
                if entity not in cells:
                    cells.append(entity)
                if loc not in cells:
                    cells.append(loc)
 
        if f.startswith("Adjacent("):
            inner = f[9:-1]
            a, b = inner.split(",", 1)
            adjacencies.setdefault(a, []).append(b)
 
        if f.startswith("MedicalPost("):
            loc = f[12:-1]
            if loc not in medical_posts:
                medical_posts.append(loc)
 
    for f in fluents:
        if f.startswith("At("):
            inner = f[3:-1]
            entity, _ = inner.split(",", 1)
            if f"HandsFree({entity})" in fluents or any(
                g.startswith(f"At({entity},") for g in fluents
            ):
                
                pass
            if f"Pickable({entity})" in fluents:
            
                pass

    for f in fluents:
        if f.startswith("HandsFree("):
            robot = f[10:-1]
            if robot not in robots:
                robots.append(robot)
 
    for f in fluents:
        if f.startswith("At("):
            inner = f[3:-1]
            entity, loc = inner.split(",", 1)
            if entity not in robots and f"Pickable({entity})" in fluents:
                
                pass
 

    supplies_list = getattr(problem, 'supplies', [])
    patients = getattr(problem, 'patients', [])
    robot = robots[0] if robots else "R"
 

    if not supplies_list or not patients:
        pickable_entities = []
        for f in fluents:
            if f.startswith("At("):
                inner = f[3:-1]
                entity, _ = inner.split(",", 1)
                if entity not in robots and f"Pickable({entity})" in fluents:
                    pickable_entities.append(entity)
 
        
        if not supplies_list:
            supplies_list = [e for e in pickable_entities if e.startswith("T") or e.startswith("t") or e.lower().startswith("supply")]
        if not patients:
            patients = [e for e in pickable_entities if e.startswith("S") or e.lower().startswith("patient") or e.lower().startswith("survivor")]
 
        #si aun no podemos distinguir, asignamos la primera mitad como suministros
        if not supplies_list and not patients and pickable_entities:
            mid = max(1, len(pickable_entities) // 2)
            supplies_list = pickable_entities[:mid]
            patients = pickable_entities[mid:]
 
    medical_post = medical_posts[0] if medical_posts else None
 
    #HLA Navigate(from, to): un refinamiento por cada celda adyacente directa
    #creamos Navigate como función que genera el HLA correspondiente
    
    navigate_hlas: dict[tuple[str, str], HLA] = {}
 
  
    for cell_a, neighbors in adjacencies.items():
        for cell_b in neighbors:
            key = (cell_a, cell_b)
            move_action = _make_move(robot, cell_a, cell_b)
            nav_hla = HLA(
                name=f"Navigate({cell_a},{cell_b})",
                refinements=[[move_action]], 
            )
            navigate_hlas[key] = nav_hla
 
    
    all_cells = list(adjacencies.keys())
 
    for cell_a in all_cells:
        for cell_b in all_cells:
            if cell_a == cell_b:
                continue
            if (cell_a, cell_b) in navigate_hlas:
                continue  
            
            for mid in adjacencies.get(cell_a, []):
                if cell_b in adjacencies.get(mid, []):
                    
                    nav_ab = HLA(name=f"Navigate({cell_a},{cell_b})", refinements=[])
                    navigate_hlas[(cell_a, cell_b)] = nav_ab
                    break
 
    
    for (cell_a, cell_b), hla_ab in navigate_hlas.items():
        if hla_ab.refinements:
            continue  
        for mid in adjacencies.get(cell_a, []):
            if (mid, cell_b) in navigate_hlas:
                nav_a_mid = navigate_hlas[(cell_a, mid)]
                nav_mid_b = navigate_hlas[(mid, cell_b)]
                hla_ab.refinements.append([nav_a_mid, nav_mid_b])
 
    
    def get_location(entity: str) -> str | None:
        for f in fluents:
            if f == f"At({entity},{{}})".format(""):
                pass
        for f in fluents:
            if f.startswith(f"At({entity},"):
                return f[len(f"At({entity},"):-1]
        return None
 
    # HLA PrepareSupplies(supplies, medical_post)
    # Refinamiento: Navigate(robot_loc, supplies_loc) + PickUp + Navigate(supplies_loc, post) + SetupSupplies

    prepare_hlas = []
 
    for s in supplies_list:
        if medical_post is None:
            continue
        s_loc = get_location(s)
        r_loc = get_location(robot)
 
        if s_loc is None or r_loc is None:
            continue
 
        refinements = []
        steps = []

        if r_loc != s_loc and (r_loc, s_loc) in navigate_hlas:
            steps.append(navigate_hlas[(r_loc, s_loc)])
        elif r_loc != s_loc:
            nav = HLA(f"Navigate({r_loc},{s_loc})", refinements=[])
            navigate_hlas[(r_loc, s_loc)] = nav
            steps.append(nav)
 
        steps.append(_make_pickup(robot, s, s_loc))
 
        if s_loc != medical_post and (s_loc, medical_post) in navigate_hlas:
            steps.append(navigate_hlas[(s_loc, medical_post)])
        elif s_loc != medical_post:
            nav = HLA(f"Navigate({s_loc},{medical_post})", refinements=[])
            navigate_hlas[(s_loc, medical_post)] = nav
            steps.append(nav)
 
        steps.append(_make_setup_supplies(robot, s, medical_post))
 
        refinements.append(steps)
 
        prepare_hla = HLA(
            name=f"PrepareSupplies({s},{medical_post})",
            refinements=refinements,
        )
        prepare_hlas.append(prepare_hla)
 
    # HLA ExtractPatient(patient, medical_post)
    # Refinamiento: Navigate(robot_loc, patient_loc) + PickUp(patient) + Navigate(patient_loc, post) + PutDown

    extract_hlas = []
 
    for p in patients:
        if medical_post is None:
            continue
        p_loc = get_location(p)
        r_loc = get_location(robot)
 
        if p_loc is None or r_loc is None:
            continue
 
        refinements = []
        steps = []
 
        if r_loc != p_loc and (r_loc, p_loc) in navigate_hlas:
            steps.append(navigate_hlas[(r_loc, p_loc)])
        elif r_loc != p_loc:
            nav = HLA(f"Navigate({r_loc},{p_loc})", refinements=[])
            navigate_hlas[(r_loc, p_loc)] = nav
            steps.append(nav)
 
        steps.append(_make_pickup(robot, p, p_loc))
 
        if p_loc != medical_post and (p_loc, medical_post) in navigate_hlas:
            steps.append(navigate_hlas[(p_loc, medical_post)])
        elif p_loc != medical_post:
            nav = HLA(f"Navigate({p_loc},{medical_post})", refinements=[])
            navigate_hlas[(p_loc, medical_post)] = nav
            steps.append(nav)
 
        steps.append(_make_putdown(robot, p, medical_post))
 
        refinements.append(steps)
 
        extract_hla = HLA(
            name=f"ExtractPatient({p},{medical_post})",
            refinements=refinements,
        )
        extract_hlas.append(extract_hla)
 
    #HLA FullRescueMission(supplies, patient, medical_post)
    #eefinamiento: PrepareSupplies + ExtractPatient + Rescue
    
    full_mission_hlas = []
 
    for (prepare_hla, extract_hla) in zip(prepare_hlas, extract_hlas):
        s = prepare_hla.name.split("(")[1].split(",")[0]
        p = extract_hla.name.split("(")[1].split(",")[0]
 
        rescue_action = _make_rescue(robot, p, medical_post)
 
        full_mission = HLA(
            name=f"FullRescueMission({s},{p},{medical_post})",
            refinements=[
                [prepare_hla, extract_hla, rescue_action]
            ],
        )
        full_mission_hlas.append(full_mission)
    return full_mission_hlas if full_mission_hlas else list(navigate_hlas.values())
    ### End of your code ###
