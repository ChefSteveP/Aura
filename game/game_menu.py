import curses  # terminal-based UI package
import time
import random


# Initialize color pairs
def initialize_colors():
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Highlighted option
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal option
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Title text
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(8, curses.COLOR_CYAN, curses.COLOR_BLACK)

# Display the title
def print_title(stdscr):
    stdscr.clear()
    ascii_art = [
        "    ____        __              __       ",
        "   / __ \\____  / /_  ___  _____/ /_____ _",
        "  / /_/ / __ \\/ __ \\/ _ \\/ ___/ __/ __ `/",
        " / _, _/ /_/ / /_/ /  __/ /  / /_/ /_/ / ",
        "/_/ |_|\\____/_.___/\\___/_/   \\__/\\__,_/  "
    ]
    for i, line in enumerate(ascii_art):
        stdscr.addstr(i, 2, line, curses.color_pair((i % 6) + 3))
    stdscr.addstr(len(ascii_art), 3, "Welcome to a new world of adventure!", curses.color_pair(3))

# Display the menu
def print_menu(stdscr, menu, selected_row):
    for idx, row in enumerate(menu):
        if idx == selected_row:
            stdscr.addstr(idx + 8, 0, f"> {row} <", curses.color_pair(1))
        else:
            stdscr.addstr(idx + 8, 0, f"  {row}  ", curses.color_pair(2))
    stdscr.refresh()

# Return to the main menu
def return_to_menu(stdscr, menu):
    print_title(stdscr)
    print_menu(stdscr, menu, 0)

# Horizontal menu with blinking
def horizontal_menu_with_blink(stdscr, text, options):
    stdscr.clear()
    options.append("Exit to Menu")
    for idx, line in enumerate(text.split("\n")):
        stdscr.addstr(idx + 2, 2, line, curses.color_pair(3))

    current_option = 0

    while True:
        option_x = 5
        for idx, option in enumerate(options):
            if idx == current_option:
                stdscr.addstr(10, option_x, f"[{option}]", curses.color_pair(1))
            else:
                stdscr.addstr(10, option_x, f" {option} ", curses.color_pair(2))
            option_x += len(option) + 4

        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_LEFT and current_option > 0:
            current_option -= 1
        elif key == curses.KEY_RIGHT and current_option < len(options) - 1:
            current_option += 1
        elif key == ord("\n"):
            if options[current_option] == "Exit to Menu":
                return None
            return current_option
        
# Function to generate a new story
def generate_story():
    stories = [
        "You find yourself in a dark forest. The trees tower above you, their branches like claws in the moonlight.\nWhat will you do?",
        "You stand at the edge of a vast desert. The sun blazes above you, and the horizon seems endless.\nWhat will you do?",
        "You wake up in a strange room with no windows and a single locked door. A faint ticking sound fills the air.\nWhat will you do?",
        "A raging river blocks your path. The current is swift, and you see no bridge in sight.\nWhat will you do?",
        "You are in a bustling marketplace filled with strange and exotic goods. A merchant offers you a mysterious box.\nWhat will you do?"
    ]
    return random.choice(stories)

# Main game loop
def main_menu(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)
    initialize_colors()

    menu = ["Start Game", "Exit"]
    current_row = 0

    return_to_menu(stdscr, menu)

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
            current_row += 1
        elif key == ord("\n"):
            if menu[current_row] == "Start Game":
                while True:
                    # Generate a new story
                    story_text = generate_story()
                    options = ["Option 1", "Option 2"]
                    selected_option = horizontal_menu_with_blink(stdscr, story_text, options)
                    
                    if selected_option is None:  # User selected "Exit to Menu"
                        return_to_menu(stdscr, menu)
                        break  # Exit the story round loop

                    # Handle selected_option here (e.g., display result of the choice)
                    result_text = f"You chose {options[selected_option]}!"
                    stdscr.clear()
                    stdscr.addstr(5, 5, result_text, curses.color_pair(3))
                    stdscr.addstr(7, 5, "Press any key to continue to the next story...", curses.color_pair(2))
                    stdscr.refresh()
                    stdscr.getch()  # Wait for user to acknowledge before the next round

            elif menu[current_row] == "Exit":
                break

        print_menu(stdscr, menu, current_row)

# Run the program
curses.wrapper(main_menu)
