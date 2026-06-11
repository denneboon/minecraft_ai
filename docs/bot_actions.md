# Bot action catalogue

Every action the bot may want, by layer. This is the roadmap for the
autonomous tree-chopping bot (and beyond) — it keeps the design honest
about what's a reusable primitive vs. a composed behaviour, so we can
implement and *separately train/tune* each one.

Status: ✅ implemented · 🟡 partial/exists-elsewhere · 🔜 planned

Skills live in `agents/skills.py` (tick-driven, emit `AgentAction`,
composable). Behaviours compose skills in a task FSM. Tool/food/block
selection goes through `control/hotbar.py` (roles), block knowledge
through `knowledge/catalog.py` + the world recogniser → `WorldMap`.

---

## 1. Look / aim
| action | status | where |
| --- | --- | --- |
| `look_at_voxel(v)` — aim crosshair at a world voxel | ✅ | `skills.LookAtVoxel` |
| `aim_angles(eye,target)` — desired yaw/pitch (pure) | ✅ | `skills.aim_angles` |
| `turn_to(yaw)` / `look(pitch)` | 🟡 | `AgentAction.look_*` |
| `scan_surroundings` — camera sweep to map blocks | 🟡 | `agents/world_explorer.py` |

## 2. Locomotion
| action | status | where |
| --- | --- | --- |
| `walk_to(target)` — A* path + follow | ✅ | `agents/walker.py` |
| `jump` / `sneak` / `sprint` / `strafe` / `stop` | ✅ | `AgentAction.movement` |
| `step_up` (1-block) — MC auto-step while walking | 🟡 | walker |
| `jump_gap(n)` — leap a 1–3 block hole | 🔜 | planner |
| `climb_ladder` / `climb_vine` | 🔜 | |
| `swim` / surface | 🔜 | |

## 3. Build / place
| action | status | where |
| --- | --- | --- |
| `pillar_up(h)` — build straight up (god-bridge mechanic) | ✅ | `skills.PillarUp` |
| `bridge(len, keep_sneak)` — extend a path; sneak = SAFE bridge, no-sneak = god-bridge | ✅ | `skills.Bridge` |
| `place_block_at(voxel, face)` — general placement | 🔜 | (generalise PillarUp/Bridge place step) |
| `staircase_up/down` | 🔜 | |
| `scaffold_to(target)` — bridge+pillar combo to reach a spot | 🔜 | |

## 4. Mine / break
| action | status | where |
| --- | --- | --- |
| `mine_block(voxel, tool_role)` — select tool, aim, attack until air | ✅ | `skills.MineBlock` |
| `clear_leaves(around v)` — break leaves to reach/expose logs | 🔜 | compose MineBlock |
| `dig_down(safe)` — mine the block below, watch for fall/lava | 🔜 | |
| `tunnel_forward(n)` — mine a 1×2 corridor | 🔜 | |
| `mine_out` — break an obstacle blocking a path (recovery) | ✅ | reuse `MineBlock` |

## 5. Inventory / hotbar
| action | status | where |
| --- | --- | --- |
| `select_role(role)` — pick the slot for sword/axe/pickaxe/blocks/food | ✅ | `skills.SelectRole`, `control/hotbar.py` |
| `select_slot(n)` | ✅ | `AgentAction.hotbar` |
| `eat()` — select food, hold use, (caller checks hunger via HUD) | ✅ | `skills.Eat` |
| `slot_for_role` / `mis_stocked` queries | ✅ | `control/hotbar.py` |
| `collect_drops` — walk over dropped items to pick them up (e.g. the logs you just mined) | 🔜 | needs item-entity detection + walk |
| `tidy_hotbar` — move mis-stocked items into reserved slots | 🔜 | needs inventory-screen drag |
| `drop_item` / `equip_armor` | 🔜 | `AgentAction` has drop hook |

## 6. Combat (later)
| action | status |
| --- | --- |
| `attack_entity` (select sword, aim, swing) | 🔜 |
| `shield_block` / `flee` | 🔜 |

## 7. Recovery / locomotion-planner tactics
The "planner" decides, given terrain, HOW to make progress or get unstuck.
| action | status | where |
| --- | --- | --- |
| `pillar_out_of_hole` — stuck below grade → pillar up | ✅ | reuse `PillarUp` |
| `mine_out` — obstacle ahead → mine through | ✅ | reuse `MineBlock` |
| `jump_over` — 1-block obstacle → jump + forward | 🔜 | planner |
| `avoid_fall` / `stop_at_edge` — don't walk off cliffs/into holes | 🔜 | planner (look-ahead in WorldMap) |
| `unstuck` — no progress → replan / wiggle | 🟡 | walker stuck-detection |
| `back_off` — retreat from a hazard | ✅ | `AgentAction.movement` |

## 8. Queries / world knowledge
| action | status | where |
| --- | --- | --- |
| `find_nearest(block_type)` — nearest known log/etc. in WorldMap | 🔜 | helper over `WorldMap.iter_blocks_in_range` |
| `is_log(id)` / role of a block | ✅ | `catalog.blocks_in_tag("logs")` |
| `reachable(voxel)` — can A* get there | 🟡 | `vision/world/pathfind.py` |

## 9. High-level behaviours (task FSM)
| behaviour | status | composes |
| --- | --- | --- |
| `goto(target)` | ✅ | walker |
| `explore` — roam to unmapped terrain, avoid falls | 🔜 | walk + planner + scan |
| `chop_tree` — path to a log, mine the trunk up, collect drops | 🔜 | find_nearest + walk + mine + collect |
| `find_and_chop_logs` — explore → spot oak log → chop → scan nearby → repeat | 🔜 | the headline goal (piece 6) |
| `collect(item)` | 🔜 | |

---

### Build order (each: offline-test the logic, then live-verify execution, then commit)
1. **Hotbar roles** ✅ — `knowledge/item_roles.py`, `control/hotbar.py`
2. **Skill primitives** ✅ — `agents/skills.py` (look/select/eat/mine/pillar/bridge)
3. Mining behaviour (tool + verify-broken + drop collection) 🔜
4. Locomotion planner (jump-over / pillar-out / mine-through / avoid-fall) 🔜
5. Exploration locomotion 🔜
6. `find_and_chop_logs` task FSM 🔜

Every skill is tick-driven and emits one `AgentAction`, so the same
interface admits a future *learned* policy (the agent loop, safety gate,
and panic-stop don't change).
