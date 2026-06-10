; God bridge (AutoHotkey) — diagonal back-strafe variant.
; The player faces a block corner (yaw at a 45° offset), looks down
; ~+66° so the crosshair lands on the SIDE face of the block-under-feet
; diagonally, then holds S+A (back + strafe-left) while spam-clicking.
; Each placement attaches to the side face → new block at the same y
; level, diagonally adjacent. Player drifts onto it, repeats.
;
; Manual setup BEFORE running:
;   1. Stand centred on a block.
;   2. Yaw must be at a corner offset (45°, 135°, -45°, or -135°).
;      This .ahk can't align yaw — use ``python tools/run_god_bridge.py``
;      which does it for you with F3 feedback.
;   3. Have blocks in the selected hotbar slot.
;
; NOTE: input macros are bannable on competitive servers (Hypixel etc.).
; Singleplayer / your own world only.

; Pitch sequence (chunked to avoid single-large-delta drops):
;   13 × -100 px UP   → guarantees the -90° clamp.
;   8  × +90  px DOWN → ~+66° at 6.7 px/°.
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R       ; 13 × -100 → -90° clamp guaranteed
Sleep, 100
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R         ; 8 × 90 → ~+66° at 6.7 px/°
Sleep, 200

Send {s down}                 ; walk backward
Send {a down}                 ; strafe left (use {d down} for mirror)

; 80 placements at MC's vanilla place-cooldown rate (5/sec).
Loop, 80
{
    Click, right
    Sleep, 200
}

Send {a up}
Send {s up}
