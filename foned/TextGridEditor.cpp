/* TextGridEditor.cpp
 *
 * Copyright (C) 1992-2022 Paul Boersma
 *
 * This code is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This code is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this work. If not, see <http://www.gnu.org/licenses/>.
 */

#include "TextGridEditor.h"
#include "EditorM.h"

Thing_implement (TextGridEditor, FunctionEditor, 0);

#include "Prefs_define.h"
#include "TextGridEditor_prefs.h"
#include "Prefs_install.h"
#include "TextGridEditor_prefs.h"
#include "Prefs_copyToInstance.h"
#include "TextGridEditor_prefs.h"

static void menu_cb_TextGridEditorHelp (TextGridEditor, EDITOR_ARGS) { HELP (U"TextGridEditor") }
static void menu_cb_AboutSpecialSymbols (TextGridEditor, EDITOR_ARGS) { HELP (U"Special symbols") }
static void menu_cb_PhoneticSymbols (TextGridEditor, EDITOR_ARGS) { HELP (U"Phonetic symbols") }
static void menu_cb_AboutTextStyles (TextGridEditor, EDITOR_ARGS) { HELP (U"Text styles") }

static void menu_cb_AreaHeights (TextGridEditor me, EDITOR_ARGS) {
	EDITOR_FORM (U"Area heights", nullptr)
		REAL (textGridHeight, U"TextGrid height", U"0.3")
		REAL (soundHeight,    U"Sound height",    U"0.4")
	EDITOR_OK
		SET_REAL (textGridHeight, my textGridAreaHeight())
		SET_REAL (soundHeight,    my soundAreaHeight())
	EDITOR_DO
		my setInstancePref_textGridHeight (Melder_clipped (0.05, textGridHeight, 0.9));
		if (my soundArea() && my soundAnalysisArea() -> hasContentToShow())
			my setInstancePref_soundHeight (Melder_clipped (0.1, soundHeight, 0.9));
		my v_distributeAreas ();
		FunctionEditor_redraw (me);
	EDITOR_END
}

void structTextGridEditor :: v_createMenus () {
	TextGridEditor_Parent :: v_createMenus ();
	EditorMenu viewMenu = Editor_addMenu (this, U"View", 0);
	EditorMenu_addCommand (viewMenu, U"Area heights...", 0, menu_cb_AreaHeights);
}

bool structTextGridEditor :: v_mouseInWideDataView (GuiDrawingArea_MouseEvent event, double x_world, double globalY_fraction) {
	if (our soundArea()) {
		const double dataBottom = our dataBottom_pxlt();
		const double dataHeight = our dataTop_pxlt() - dataBottom;
		const double y_px = dataBottom + globalY_fraction * dataHeight;
		const bool showAnalysis = our soundAnalysisArea() -> hasContentToShow();
		const double textGridHeight = our textGridAreaHeight();
		const double textGridTop_px = dataBottom + textGridHeight * dataHeight;
		const double soundBottom_px = dataBottom +
				( showAnalysis ? 1.0 - our soundAreaHeight() : textGridHeight ) * dataHeight;
		if (event -> isClick ()) {
			our d_draggedBoundary = 0;
			if (y_px >= textGridTop_px - 25.0 && y_px <= textGridTop_px + 5.0)
				our d_draggedBoundary = 1;
			else if (showAnalysis && y_px >= soundBottom_px - 25.0 && y_px <= soundBottom_px + 5.0)
				our d_draggedBoundary = 2;
			if (our d_draggedBoundary != 0) {
				our setInstancePref_textGridHeight (our textGridAreaHeight());   // freeze the automatic layout
				if (showAnalysis)
					our setInstancePref_soundHeight (our soundAreaHeight());
				return true;
			}
		} else if (our d_draggedBoundary != 0) {
			if (event -> isDrag ()) {
				if (our d_draggedBoundary == 1) {
					our setInstancePref_textGridHeight (Melder_clipped (0.05, globalY_fraction, 0.9));
				} else {
					our setInstancePref_soundHeight (Melder_clipped (0.1, 1.0 - globalY_fraction, 0.9));
				}
				our v_distributeAreas ();
				FunctionEditor_redraw (this);
			} else if (event -> isDrop ()) {
				our d_draggedBoundary = 0;
			}
			return true;
		}
	}
	return structFunctionEditor :: v_mouseInWideDataView (event, x_world, globalY_fraction);
}

void structTextGridEditor :: v_createMenuItems_help (EditorMenu menu) {
	TextGridEditor_Parent :: v_createMenuItems_help (menu);
	EditorMenu_addCommand (menu, U"TextGridEditor help", '?', menu_cb_TextGridEditorHelp);
	EditorMenu_addCommand (menu, U"About special symbols", 0, menu_cb_AboutSpecialSymbols);
	EditorMenu_addCommand (menu, U"Phonetic symbols", 0, menu_cb_PhoneticSymbols);
	EditorMenu_addCommand (menu, U"About text styles", 0, menu_cb_AboutTextStyles);
}

autoTextGridEditor TextGridEditor_create (conststring32 title, TextGrid textGrid,
	SampledXY optionalSoundOrLongSound, SpellingChecker spellingChecker, conststring32 callbackSocket)
{
	try {
		autoTextGridEditor me = Thing_new (TextGridEditor);
		my textGridArea() = TextGridArea_create (true, nullptr, me.get());
		if (optionalSoundOrLongSound) {
			Melder_assert (optionalSoundOrLongSound -> ny > 0);
			if (Thing_isa (optionalSoundOrLongSound, classSound))
				my soundArea() = SoundArea_create (false, static_cast <Sound> (optionalSoundOrLongSound), me.get());
			else
				my soundArea() = LongSoundArea_create (false, static_cast <LongSound> (optionalSoundOrLongSound), me.get());
			my soundAnalysisArea() = SoundAnalysisArea_create (false, nullptr, me.get());
			my textGridArea() -> borrowedSoundArea = my soundArea().get();
			my textGridArea() -> borrowedSoundAnalysisArea = my soundAnalysisArea().get();
		}
		my textGridArea() -> spellingChecker = spellingChecker;
		my callbackSocket = Melder_dup (callbackSocket);
		FunctionEditor_init (me.get(), title, textGrid);

		Melder_assert (isdefined (my startSelection));   // precondition of v_updateText()
		my v_updateText ();   // to reflect changed tier selection AND to get first text; BUG: should not be needed
		if (spellingChecker)
			GuiText_setSelection (my textArea, 0, 0);
		if (optionalSoundOrLongSound &&
			optionalSoundOrLongSound -> xmin == 0.0 &&
			my textGrid() -> xmin != 0.0 &&
			my textGrid() -> xmax > optionalSoundOrLongSound -> xmax
		)
			Melder_warning (U"The time domain of the TextGrid (starting at ",
				Melder_fixed (my textGrid() -> xmin, 6), U" seconds) does not overlap with that of the sound "
				U"(which starts at 0 seconds).\nIf you want to repair this, you can select the TextGrid "
				U"and choose “Shift times to...” from the Modify menu "
				U"to shift the starting time of the TextGrid to zero.");
		return me;
	} catch (MelderError) {
		Melder_throw (U"TextGrid window not created.");
	}
}

/* End of file TextGridEditor.cpp */
