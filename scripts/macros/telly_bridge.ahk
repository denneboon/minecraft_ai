; Telly (tele) bridge — fast backward sneak-bridge (AutoHotkey form).
; Same mechanics as telly_bridge.mcs: look down, hold Shift (sneak) +
; S (back), spam right-click to place. Face AWAY from the bridge
; direction, blocks in the selected hotbar slot, then run.
;
; NOTE: automation macros violate competitive-server rules (Hypixel
; etc.). Singleplayer / your own world only.
; Self-levelling pitch setup (independent of starting view):
;   push UP past the -90 clamp, then DOWN to the telly angle (~45°).
; Single large vertical SendInput deltas get dropped, so all in chunks.
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R
Sleep, 20
MouseMove, 0, -100, , R      ; ≈ -75° accumulated → pitch hits -90° clamp
Sleep, 80
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
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R
Sleep, 20
MouseMove, 0, 90, , R        ; 10 × 90 ≈ 135° down → final pitch ≈ +45° (telly)
Sleep, 150
Send {Shift down}        ; sneak
Send {s down}            ; walk backward
Loop, 30
{
    Click, right         ; place
    Sleep, 90            ; fast telly cadence
}
Send {s up}
Send {Shift up}
