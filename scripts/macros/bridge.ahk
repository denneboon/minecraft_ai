; Sneak-bridge backward (AutoHotkey subset).
; The runner understands Send {key up/down}, Sleep, Click, MouseMove (R),
; and Loop { ... }. Hotkey labels / other AHK directives are ignored, so
; you can paste a macro that's bound to a key and it'll still play.
MouseMove, 0, 150, , R    ; look down toward the edge
Send {Shift down}         ; sneak
Send {s down}             ; walk backward
Loop, 15
{
    Click, right          ; place a block
    Sleep, 260
}
Send {s up}
Send {Shift up}
