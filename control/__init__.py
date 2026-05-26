"""
Hardware-input layer: the AI's *only* path to the operating system.

Every key press, mouse movement, and click that reaches Minecraft
flows through one of:

* :class:`control.keyboard.Keyboard` — pynput-backed key state machine
  with debounce, rate-limiting, hotbar tick-cooldown, and a verbose
  diagnostic trace.
* :class:`control.mouse.Mouse` — Win32 ``SendInput`` (with a pynput
  fallback) plus a 240 Hz background velocity worker for smooth,
  humanlike camera motion.
* :class:`control.action_wrapper.ActionWrapper` — single dispatcher
  the agent loop calls; translates a high-level action name into the
  right combination of keyboard / mouse events.

Safety:

* :class:`control.safety.Safety` watches Minecraft's window focus and
  the panic hotkey, and is the only thing allowed to set the gate's
  state.
* :class:`control.input_gate.InputGate` is the *single source of truth*
  for "may we send input right now?". Every press / move / click in
  this package consults it before emitting; releases bypass the gate
  so stuck keys can never outlive a focus-loss event.

The public types and helpers live in their respective modules; this
package's ``__init__`` deliberately re-exports nothing so that
import-time side effects stay zero.
"""
