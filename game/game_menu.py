import curses  # terminal-based UI package


def main(stdscr):  
    curses.curs_set(0)  # 0 hides the cursor, making the interface cleaner
    stdscr.keypad(True)  # Enable special keys (like arrow keys) to be recognized

    # Define menu options
    menu = ["Start Game", "Settings", "Exit"]  # A list of options for the user to choose from
    current_row = 0  # Tracks the currently selected menu item, initially the first one

    # Function to display the menu with the selected option highlighted
    def print_menu(stdscr, selected_row):
        stdscr.clear()  # Clear the screen to avoid overlapping text
        for idx, row in enumerate(menu):
            # Highlight the selected row using a color pair
            if idx == selected_row:
                stdscr.addstr(idx + 2, 2, f"> {row}", curses.color_pair(1))
            else:
                stdscr.addstr(idx + 2, 2, row)  # Display other rows normally
        stdscr.refresh()  # Refresh the screen to show updated content

    # Setup color pairs for the menu
    curses.start_color()  # Initialize color functionality in curses
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Define color pair 1: white background, black text

    # Display the initial menu
    print_menu(stdscr, current_row)

    # Main loop to handle user input and navigation
    while True:
        key = stdscr.getch()  # Wait for user input (key press)

        # Navigate up in the menu, ensuring the selection doesn't go out of bounds
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        # Navigate down in the menu, ensuring the selection doesn't go out of bounds
        elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
            current_row += 1
        # Handle the Enter key to confirm a selection
        elif key == ord("\n"):  # ord("\n") detects the Enter key
            # Display the selected option at the bottom of the menu
            stdscr.addstr(len(menu) + 3, 2, f"You selected '{menu[current_row]}'")
            stdscr.refresh()  # Refresh the screen to show the message
            stdscr.getch()  # Wait for the user to press a key before continuing

            # If the "Exit" option is selected, break the loop and exit
            if menu[current_row] == "Exit":
                break

        # Update the menu display with the new selected row
        print_menu(stdscr, current_row)

# The curses wrapper ensures proper initialization and cleanup of the terminal
curses.wrapper(main)
