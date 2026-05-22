# tools/safety_smoketest.py
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from control.safety import Safety, SafetyConfig
from utils.focus import activate_minecraft


s = Safety(SafetyConfig(
    log_actions=True,
    log_focus_events=True,
    print_status_table=False,
    auto_stop_on_focus_loss=False,
    focus_loss_debounce_ms=250,
    startup_focus_grace_ms=800,
    debug_focus_trace=True,   # <--- add this
))
s.start()

activate_minecraft()
print("Waiting for focus gate to allow input...")
t0 = time.time()
while not s.allow_input() and time.time() - t0 < 5:
    time.sleep(0.05)

print("allow_input() =", s.allow_input(), "(expected True if Minecraft focused)")

# Keep it alive for observation without needing to Alt-Tab
print("Observing for 5 seconds. Try Alt-Tab once and back, watch allow_input() flip.")
t1 = time.time()
while time.time() - t1 < 5:
    print("allow_input() currently:", s.allow_input())
    time.sleep(0.5)

s.stop()
print("Done.")