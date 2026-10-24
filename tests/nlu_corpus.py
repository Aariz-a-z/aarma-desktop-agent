"""The paraphrase and false-positive corpora.

Kept as data in its own module because two different consumers need it: the
test-suite asserts on it, and the evaluation harness measures accuracy and
latency over it.  A corpus embedded in test functions can only be asserted, not
measured.

Two halves, and the second matters more:

:data:`PARAPHRASES` -- at least ten natural phrasings per major action, the
thing the phase is *for*.

:data:`FALSE_POSITIVES` -- sentences that mention a capability without
requesting it.  A resolver that scores well on the first half and badly on this
one is worse than no resolver at all, because a missed command costs a repeat
and a wrongly-executed one costs whatever it did.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: (sentence, expected action, expected entities-subset).
#: The entity dict is checked as a subset, so a case can assert the query
#: without pinning every field the extractor happens to fill.
Case = Tuple[str, str, Dict[str, str]]

#: Context each group is interpreted in: None, "browser" or "editor".
Group = Tuple[str, Optional[str], List[Case]]


PARAPHRASES: List[Group] = [
    # ------------------------------------------------------- site search
    ("site search", None, [
        ("Open YouTube and search for Python tutorial.", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("Open Python tutorial on YouTube.", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("Search YouTube for a Python tutorial.", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("Go to YouTube and find Python tutorials.", "search_site",
         {"site": "youtube", "query": "python tutorials"}),
        ("Can you find me a Python tutorial on YouTube?", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("I want to watch a Python tutorial on YouTube.", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("Find a Python tutorial on YouTube for me.", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("Take me to YouTube and search Python tutorial.", "search_site",
         {"site": "youtube", "query": "python tutorial"}),
        ("Show me Python tutorials on YouTube.", "search_site",
         {"site": "youtube", "query": "python tutorials"}),
        ("look up python tutorials on you tube", "search_site",
         {"site": "youtube", "query": "python tutorials"}),
        ("search github for a json parser", "search_site",
         {"site": "github", "query": "json parser"}),
    ]),

    # ------------------------------------------------------------ volume
    ("volume down", None, [
        ("lower the sound", "volume_down", {}),
        ("turn the sound down", "volume_down", {}),
        ("make it quieter", "volume_down", {}),
        ("reduce volume", "volume_down", {}),
        ("volume down", "volume_down", {}),
        ("please lower my volume", "volume_down", {}),
        ("can you make the volume a little lower?", "volume_down",
         {"amount": "small"}),
        ("turn my volume down", "volume_down", {}),
        ("decrease the audio", "volume_down", {}),
        ("make the sound softer", "volume_down", {}),
        ("volum down", "volume_down", {}),
    ]),
    ("volume up", None, [
        ("raise the volume", "volume_up", {}),
        ("turn the sound up", "volume_up", {}),
        ("make it louder", "volume_up", {}),
        ("increase volume", "volume_up", {}),
        ("volume up", "volume_up", {}),
        ("turn my volume up", "volume_up", {}),
        ("boost the audio", "volume_up", {}),
        ("crank it", "volume_up", {}),
        ("louder please", "volume_up", {}),
        ("can you turn the music up", "volume_up", {}),
    ]),

    # ------------------------------------------------------- applications
    ("applications", None, [
        ("Open the command line.", "open_terminal", {"application": "cmd"}),
        ("Start the Windows command prompt.", "open_terminal",
         {"application": "cmd"}),
        ("open cmd", "open_terminal", {}),
        ("bring up powershell", "open_terminal", {"application": "powershell"}),
        ("Launch the terminal.", "open_terminal",
         {"application": "windowsterminal"}),
        ("fire up power shell", "open_terminal", {"application": "powershell"}),
        ("Start the calculator.", "open_application",
         {"application": "calculator"}),
        ("open notepad", "open_application", {"application": "notepad"}),
        ("launch chrome", "open_application", {"application": "chrome"}),
        ("opn chrome", "open_application", {"application": "chrome"}),
        ("pull up task manager", "open_application",
         {"application": "taskmanager"}),
    ]),

    # ------------------------------------------------------------ folders
    ("folders", None, [
        ("Open my downloads.", "open_folder", {"folder": "downloads"}),
        ("open downloads", "open_folder", {"folder": "downloads"}),
        ("show me my documents", "open_folder", {"folder": "documents"}),
        ("take me to my desktop", "open_folder", {"folder": "desktop"}),
        ("open the downloads folder", "open_folder", {"folder": "downloads"}),
        ("bring up my pictures", "open_folder", {"folder": "pictures"}),
        ("go to my documents", "open_folder", {"folder": "documents"}),
        ("open my download folder", "open_folder", {"folder": "downloads"}),
        ("launch my videos", "open_folder", {"folder": "videos"}),
        ("open my music", "open_folder", {"folder": "music"}),
    ]),

    # ----------------------------------------------------- system health
    ("system health", None, [
        ("check system health", "system_health", {}),
        ("how is my PC doing", "system_health", {}),
        ("how is my computer doing", "system_health", {}),
        ("how is my laptop doing", "system_health", {}),
        ("how much RAM am I using", "memory_usage", {}),
        ("check my memory", "memory_usage", {}),
        ("what is my cpu doing", "cpu_usage", {}),
        ("how much disk space do I have", "disk_usage", {}),
        ("check the battery", "battery_status", {}),
        ("am I charging", "battery_status", {}),
    ]),

    # -------------------------------------------------------- screenshots
    ("screenshots", None, [
        ("take a screenshot", "take_screenshot", {}),
        ("capture my screen", "take_screenshot", {}),
        ("grab a screenshot", "take_screenshot", {}),
        ("screenshot", "take_screenshot", {}),
        ("screnshot", "take_screenshot", {}),
        ("take a picture of my screen", "take_screenshot", {}),
        ("snap the screen", "take_screenshot", {}),
        ("save a screenshot", "take_screenshot", {}),
        ("capture the screen", "take_screenshot", {}),
        ("can you screenshot this screen", "take_screenshot", {}),
    ]),

    # -------------------------------------------------- browser, in front
    ("browser tabs", "browser", [
        ("close this browser page", "close_tab", {}),
        ("shut this tab", "close_tab", {}),
        ("get rid of this tab", "close_tab", {}),
        ("close the current tab", "close_tab", {}),
        ("close this", "close_tab", {}),
        ("open a new tab", "new_tab", {}),
        ("open another tab", "new_tab", {}),
        ("give me a new tab", "new_tab", {}),
        ("go back", "go_back", {}),
        ("go forward", "go_forward", {}),
        ("refresh this page", "refresh_page", {}),
        ("reload the page", "refresh_page", {}),
    ]),

    # ---------------------------------------------------------- reminders
    ("reminders", None, [
        ("remind me at 6 PM to submit my project", "create_reminder", {}),
        ("set a reminder for tomorrow at 9", "create_reminder", {}),
        ("alert me in 30 minutes", "create_reminder", {}),
        ("notify me tonight at 8", "create_reminder", {}),
        ("remind me tomorrow morning", "create_reminder", {}),
        ("ping me in an hour", "create_reminder", {}),
        ("nudge me at noon", "create_reminder", {}),
        ("remind me this evening", "create_reminder", {}),
        ("set a reminder for friday at 3", "create_reminder", {}),
        ("remind me next monday at 2pm", "create_reminder", {}),
    ]),

    # ----------------------------------------------------------- calendar
    ("calendar", None, [
        ("schedule a meeting tomorrow at 10 AM", "create_event", {}),
        ("book a meeting friday at 3", "create_event", {}),
        ("put a meeting on my calendar tomorrow at 2", "create_event", {}),
        ("add an event tomorrow at 9", "create_event", {}),
        ("arrange a call tomorrow at 4", "create_event", {}),
        ("set up a meeting next monday at 11", "create_event", {}),
        ("pencil in a meeting tonight at 7", "create_event", {}),
        ("block out tomorrow at 1", "create_event", {}),
        ("create an event tomorrow at 10", "create_event", {}),
        ("schedule a call this evening", "create_event", {}),
    ]),

    # -------------------------------------------------------------- power
    # Resolution only. Every one of these still meets SYSTEM_CONTROL and a
    # confirmation downstream; that is asserted separately.
    ("power", None, [
        ("shut down the computer", "shutdown_computer", {}),
        ("Could you maybe turn the computer off?", "shutdown_computer", {}),
        ("power off my pc", "shutdown_computer", {}),
        ("turn off the computer", "shutdown_computer", {}),
        ("restart my computer", "restart_computer", {}),
        ("reboot the pc", "restart_computer", {}),
        ("lock my pc", "lock_screen", {}),
        ("lock the computer", "lock_screen", {}),
        ("sign me out", "sign_out", {}),
        ("put the computer to sleep", "sleep_computer", {}),
    ]),

    # ------------------------------------------------------------ weather
    ("weather", None, [
        ("what's the weather", "get_weather", {}),
        ("what is the weather today", "get_weather", {}),
        ("how is the weather", "get_weather", {}),
        ("what's the weather in Bengaluru", "get_weather", {"site": ""}),
        ("will it rain tomorrow", "will_it_rain", {}),
        ("is it going to rain today", "will_it_rain", {}),
        ("do I need an umbrella", "will_it_rain", {}),
    ]),

    # ------------------------------------------------------------ windows
    ("windows", "editor", [
        ("close this", "close_window", {}),
        ("minimize this window", "minimize_window", {}),
        ("maximize this window", "maximize_window", {}),
    ]),
]


#: Sentences that name a capability without asking for it.  Every one must
#: produce no action at all.
FALSE_POSITIVES: List[Tuple[str, Optional[str]]] = [
    # -- reported speech ------------------------------------------------
    ("My friend said close the tab.", "browser"),
    ("She told me to shut down the computer.", None),
    ("My boss asked me to delete that file.", None),
    ("My friend uses GitHub.", None),
    ("They said open Chrome.", None),

    # -- hypothetical ----------------------------------------------------
    ("I'm thinking about shutting down my computer.", None),
    ("I might restart my PC later.", None),
    ("I was wondering if I should delete this.", None),
    ("Maybe I should lower the volume.", None),

    # -- instructional ---------------------------------------------------
    ("Can you tell me how to delete a file?", None),
    ("How do I take a screenshot?", None),
    ("How do you open the command prompt?", None),
    ("Tell me how to close a tab.", "browser"),

    # -- negation --------------------------------------------------------
    ("I don't want you to open Chrome.", None),
    ("Don't close this tab.", "browser"),
    ("Don't send that email.", None),
    ("Do not shut down the computer.", None),
    ("Never delete my documents.", None),

    # -- satisfied state -------------------------------------------------
    ("My volume is already fine.", None),
    ("The tab is already closed.", "browser"),
    ("My computer is fine.", None),

    # -- out of domain ---------------------------------------------------
    ("I want you to fly a drone.", None),
    ("Order me a pizza.", None),
    ("Book me a flight to Tokyo.", None),
    ("Translate this into Klingon for me.", None),
]


#: Sentences that are genuinely two-ways ambiguous and must produce a question
#: rather than an action.
AMBIGUOUS: List[Tuple[str, Optional[str]]] = [
    ("open my editor", None),
    ("open my browser", None),
]


def paraphrase_cases() -> List[Tuple[str, str, Dict[str, str], Optional[str]]]:
    """Flatten the groups into ``(sentence, action, entities, context)``."""
    flat = []
    for _group, context, cases in PARAPHRASES:
        for sentence, action, entities in cases:
            flat.append((sentence, action, entities, context))
    return flat


def total_cases() -> int:
    return len(paraphrase_cases()) + len(FALSE_POSITIVES) + len(AMBIGUOUS)


__all__ = ["AMBIGUOUS", "FALSE_POSITIVES", "PARAPHRASES", "paraphrase_cases",
           "total_cases"]
