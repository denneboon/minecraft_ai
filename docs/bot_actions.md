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

## 9. High-level behaviours (agents — `agents/treechop.py`)
| behaviour | status | composes |
| --- | --- | --- |
| `goto(target)` | ✅ | `NavigateTo` (A* route + reactive follow) |
| `find_and_chop_logs` (`--agent treechop`) | ✅ | find → navigate → ChopTrunk → collect → explore |
| `harvest` (`--agent harvest`) — gather ANY block type | ✅ | same FSM, plain MineBlock + a block predicate |
| `planner` (`--agent planner`) — run an ordered multi-task plan | ✅ | sequences behaviours; advances on goal/DONE |
| `explore` — roam to unmapped terrain, avoid falls | ✅ | walk + scan, fan headings |
| `collect(item)` — walk over drops | 🟡 | incidental (auto-pickup while standing); no item-entity vision yet |

---

### Roadmap (each: offline-test the logic, live-verify execution, commit)
1. **Hotbar roles** ✅ · 2. **Skill primitives** ✅ · 3. **Mining** ✅ (F3 aim→freeze latch, reach-gated, species-agnostic)
4. **Locomotion planner** ✅ (jump-over / pillar-out / mine-through / avoid-fall + A* `NavigateTo`)
5. **Exploration** ✅ · 6. **find_and_chop_logs FSM** ✅
7. **T2 Data** ✅ — `brain/episode_logger.py` (per-run obs/action/outcome JSONL) + `tools/export_dataset.py` (→ ML rows)
8. **T2 Generalisation** ✅ — `harvest` (any block via `mine_action` hook)
9. **T3 Goal/planner** ✅ — `PlannerAgent` sequences tasks toward gathered-count goals
— Next: closed-loop INVENTORY verify (vs gathered-count proxy); perception self-teach (species/ores); behaviour-cloning on the episode dataset.

Every skill is tick-driven and emits one `AgentAction`, so the same
interface admits a future *learned* policy (the agent loop, safety gate,
and panic-stop don't change).
