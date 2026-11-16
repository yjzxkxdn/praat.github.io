/* SoundEditor.cpp
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

#include "SoundEditor.h"
#include "EditorM.h"

Thing_implement (SoundEditor, FunctionEditor, 0);

#include "Prefs_define.h"
#include "SoundEditor_prefs.h"
#include "Prefs_install.h"
#include "SoundEditor_prefs.h"
#include "Prefs_copyToInstance.h"
#include "SoundEditor_prefs.h"

static void menu_cb_SoundEditorHelp (SoundEditor, EDITOR_ARGS) { Melder_help (U"SoundEditor"); }
static void menu_cb_LongSoundEditorHelp (SoundEditor, EDITOR_ARGS) { Melder_help (U"LongSoundEditor"); }

static void menu_cb_AreaHeights (SoundEditor me, EDITOR_ARGS) {
	EDITOR_FORM (U"Area heights", nullptr)
		REAL (soundHeight, U"Sound height", U"0.5")
	EDITOR_OK
		SET_REAL (soundHeight, my soundAreaHeight())
	EDITOR_DO
		my setInstancePref_soundHeight (Melder_clipped (0.1, soundHeight, 0.9));
		my v_distributeAreas ();
		FunctionEditor_redraw (me);
	EDITOR_END
}

void structSoundEditor :: v_createMenus () {
	structFunctionEditor :: v_createMenus ();
	EditorMenu viewMenu = Editor_addMenu (this, U"View", 0);
	EditorMenu_addCommand (viewMenu, U"Area heights...", 0, menu_cb_AreaHeights);
}

bool structSoundEditor :: v_mouseInWideDataView (GuiDrawingArea_MouseEvent event, double x_world, double globalY_fraction) {
	if (our soundAnalysisArea() -> hasContentToShow ()) {
		const double dataBottom = our dataBottom_pxlt();
		const double dataHeight = our dataTop_pxlt() - dataBottom;
		const double y_px = dataBottom + globalY_fraction * dataHeight;
		const double boundary_px = dataBottom + our soundAreaHeight() * dataHeight;
		if (event -> isClick ()) {
			our d_draggingBoundary = ( y_px >= boundary_px - 25.0 && y_px <= boundary_px + 5.0 );
			if (our d_draggingBoundary) {
				our setInstancePref_soundHeight (our soundAreaHeight());   // freeze the automatic layout
				return true;
			}
		} else if (our d_draggingBoundary) {
			if (event -> isDrag ()) {
				our setInstancePref_soundHeight (Melder_clipped (0.1, globalY_fraction, 0.9));
				our v_distributeAreas ();
				FunctionEditor_redraw (this);
			} else if (event -> isDrop ()) {
				our d_draggingBoundary = false;
			}
			return true;
		}
	}
	return structFunctionEditor :: v_mouseInWideDataView (event, x_world, globalY_fraction);
}

void structSoundEditor :: v_createMenuItems_help (EditorMenu menu) {
	structFunctionEditor :: v_createMenuItems_help (menu);
	EditorMenu_addCommand (menu, U"SoundEditor help", '?', menu_cb_SoundEditorHelp);
	EditorMenu_addCommand (menu, U"LongSoundEditor help", 0, menu_cb_LongSoundEditorHelp);
	// BUG: add help on Sound area and Sound analysis area
}

autoSoundEditor SoundEditor_create (conststring32 title, SampledXY soundOrLongSound) {
	Melder_assert (soundOrLongSound);
	Melder_assert (soundOrLongSound -> ny > 0);
	try {
		autoSoundEditor me = Thing_new (SoundEditor);
		if (Thing_isa (soundOrLongSound, classSound))
			my soundArea() = SoundArea_create (true, nullptr, me.get());
		else
			my soundArea() = LongSoundArea_create (false, nullptr, me.get());
		my soundAnalysisArea() = SoundAnalysisArea_create (false, nullptr, me.get());
		FunctionEditor_init (me.get(), title, soundOrLongSound);
		return me;
	} catch (MelderError) {
		Melder_throw (U"Sound window not created.");
	}
}

/* End of file SoundEditor.cpp */
