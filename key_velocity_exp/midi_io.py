"""MIDI input from the digital piano (USB-MIDI), timestamped with time.monotonic().

Digital pianos send Note On (velocity 1-127) / Note Off (or Note On with velocity 0) on channel 0.
``kve-midi --list`` prints the port names (a Yamaha P-45 shows up as
``Digital Piano:Digital Piano MIDI 1 20:0``); ``kve-midi`` listens and prints every event.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from key_velocity_exp.common import MidiEvent, now

# tried in order when no port substring is given (add your piano's name here or pass --midi-port)
PREFERRED_PORT_SUBSTRINGS = ("digital piano", "p-45", "p45", "yamaha", "kawai", "roland", "casio", "korg", "nord", "piano")


class MidiListener:
    def __init__(self, port_substring: Optional[str] = None, on_event: Optional[Callable[[MidiEvent], None]] = None):
        import mido

        self._mido = mido
        self.port_name = self._resolve_port(port_substring)
        self._lock = threading.Lock()
        self._events: list[MidiEvent] = []
        self._on_event = on_event
        self._port = mido.open_input(self.port_name, callback=self._callback)

    @staticmethod
    def list_ports() -> list[str]:
        try:
            import mido

            return mido.get_input_names()
        except ImportError as e:  # python-rtmidi is a compiled extension: on Linux it needs the ALSA runtime
            raise RuntimeError(
                f"cannot load the MIDI backend ({e}). On Debian/Ubuntu install it with: sudo apt install libasound2 (then re-run); on other systems check that python-rtmidi imports."
            ) from e

    def _resolve_port(self, sub: Optional[str]) -> str:
        names = self.list_ports()
        if sub is None:
            cands = [n for n in names if "through" not in n.lower() and "pipewire" not in n.lower()]
            for pref in PREFERRED_PORT_SUBSTRINGS:
                for n in cands:
                    if pref in n.lower():
                        return n
            if len(cands) == 1:
                return cands[0]
            raise RuntimeError(f"Cannot auto-select a MIDI input port. Available: {names}. Pass --midi-port <substring>.")
        for n in names:
            if sub.lower() in n.lower():
                return n
        raise RuntimeError(f"No MIDI input port containing '{sub}'. Available: {names}")

    def _callback(self, msg) -> None:
        t = now()
        if msg.type == "note_on" and msg.velocity > 0:
            ev = MidiEvent(t=t, kind="note_on", note=msg.note, velocity=msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            ev = MidiEvent(t=t, kind="note_off", note=msg.note, velocity=0)
        else:
            return
        with self._lock:
            self._events.append(ev)
        if self._on_event is not None:
            self._on_event(ev)

    def drain(self) -> list[MidiEvent]:
        """Return and clear all events collected so far."""
        with self._lock:
            ev, self._events = self._events, []
        return ev

    def events_between(self, t0: float, t1: float, clear: bool = False) -> list[MidiEvent]:
        with self._lock:
            sel = [e for e in self._events if t0 <= e.t <= t1]
            if clear:
                self._events = [e for e in self._events if not (t0 <= e.t <= t1)]
        return sel

    def close(self) -> None:
        self._port.close()


class SimPiano:
    """Virtual two-contact key for dry runs.

    A key has two switches at depths ``d1 < d2`` below contact. Note-on fires when the fingertip passes
    ``d2``; velocity is a function of the time taken to travel from ``d1`` to ``d2`` (this is how the
    real P-45 GHS action measures velocity). Note-off fires when the key is released above ``d1``.
    The event is reported with a fixed transmission latency.
    """

    def __init__(self, d1: float = 0.0030, d2: float = 0.0080, latency: float = 0.006, curve: str = "log"):
        self.d1, self.d2, self.latency, self.curve = d1, d2, latency, curve
        self._t1 = {}  # note -> time first switch closed
        self._down = set()
        self.events: list[MidiEvent] = []
        self._lock = threading.Lock()

    def velocity_from_dt(self, dt: float) -> int:
        # mean speed between the switches in m/s
        v = (self.d2 - self.d1) / max(dt, 1e-4)
        # pretend P-45 "medium": ~15 at 0.02 m/s ... ~110 at 0.5 m/s, log-linear
        import numpy as np

        vel = 15 + (110 - 15) * (np.log(v) - np.log(0.02)) / (np.log(0.5) - np.log(0.02))
        return int(np.clip(round(vel), 1, 127))

    def update(self, note: int, depth: float, t: float) -> None:
        if depth >= self.d1 and note not in self._t1 and note not in self._down:
            self._t1[note] = t
        if depth >= self.d2 and note in self._t1 and note not in self._down:
            vel = self.velocity_from_dt(t - self._t1.pop(note))
            self._down.add(note)
            self._emit(MidiEvent(t=t + self.latency, kind="note_on", note=note, velocity=vel))
        if depth < self.d1:
            self._t1.pop(note, None)
            if note in self._down:
                self._down.discard(note)
                self._emit(MidiEvent(t=t + self.latency, kind="note_off", note=note, velocity=0))

    def _emit(self, ev: MidiEvent) -> None:
        with self._lock:
            self.events.append(ev)

    # same interface as MidiListener
    def events_between(self, t0: float, t1: float, clear: bool = False) -> list[MidiEvent]:
        with self._lock:
            sel = [e for e in self.events if t0 <= e.t <= t1]
            if clear:
                self.events = [e for e in self.events if not (t0 <= e.t <= t1)]
        return sel

    def drain(self) -> list[MidiEvent]:
        with self._lock:
            ev, self.events = self.events, []
        return ev

    def close(self) -> None:
        pass


def main() -> None:
    import argparse
    import os

    from key_velocity_exp.common import MIDI_PORT_ENV, midi_to_name

    p = argparse.ArgumentParser(description="List MIDI input ports or listen to the piano and print every note event.")
    p.add_argument("--list", action="store_true", help="only list the input ports")
    p.add_argument("--midi-port", default=os.environ.get(MIDI_PORT_ENV), help=f"substring of the port name (default: ${MIDI_PORT_ENV}, else auto-detect)")
    a = p.parse_args()
    print("MIDI input ports:", MidiListener.list_ports())
    if a.list:
        return
    lst = MidiListener(a.midi_port)
    print(f"Listening on '{lst.port_name}' - play some keys, Ctrl-C to stop")
    try:
        while True:
            time.sleep(0.2)
            for e in lst.drain():
                print(f"{e.t:.4f} {e.kind:8s} {midi_to_name(e.note):4s} note={e.note:3d} vel={e.velocity}")
    except KeyboardInterrupt:
        lst.close()


if __name__ == "__main__":
    main()
