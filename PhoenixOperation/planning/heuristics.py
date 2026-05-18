from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    unsatisfied = list(goal - state)
    if not unsatisfied:
        return 0

    goal_index = {fluent: index for index, fluent in enumerate(unsatisfied)}
    full_mask = (1 << len(unsatisfied)) - 1
    cover_masks = []

    for action in get_all_groundings(domain, objects):
        mask = 0
        for fluent in action.add_list & goal:
            if fluent in goal_index:
                mask |= 1 << goal_index[fluent]
        if mask:
            cover_masks.append(mask)

    distances = [float("inf")] * (full_mask + 1)
    distances[0] = 0
    for mask in range(full_mask + 1):
        if distances[mask] == float("inf"):
            continue
        for cover_mask in cover_masks:
            next_mask = mask | cover_mask
            distances[next_mask] = min(distances[next_mask], distances[mask] + 1)

    return distances[full_mask]

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    relaxed_state = frozenset(state)
    if goal.issubset(relaxed_state):
        return 0

    steps = 0
    actions = get_all_groundings(domain, objects)

    while not goal.issubset(relaxed_state):
        applicable_actions = [
            action
            for action in actions
            if action.precond_pos.issubset(relaxed_state)
            and action.precond_neg.isdisjoint(relaxed_state)
        ]
        improving_actions = [
            action for action in applicable_actions if action.add_list - relaxed_state
        ]
        if not improving_actions:
            return float("inf")

        best_action = max(
            improving_actions,
            key=lambda action: (
                len(goal & (relaxed_state | action.add_list)),
                len(action.add_list - relaxed_state),
                action.name,
            ),
        )
        relaxed_state = frozenset(relaxed_state | best_action.add_list)
        steps += 1

    return steps

    ### End of your code ###
