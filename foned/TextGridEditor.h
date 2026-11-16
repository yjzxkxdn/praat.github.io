#ifndef _TextGridEditor_h_
#define _TextGridEditor_h_
/* TextGridEditor.h
 *
 * Copyright (C) 1992-2005,2007-2022 Paul Boersma
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

#include "FunctionEditor.h"
#include "TextGridArea.h"
#include "LongSoundArea.h"
#include "SoundAnalysisArea.h"

Thing_define (TextGridEditor, FunctionEditor) {
	DEFINE_FunctionArea (1, TextGridArea, textGridArea)
	DEFINE_FunctionArea (2, SoundArea, soundArea)
	DEFINE_FunctionArea (3, SoundAnalysisArea, soundAnalysisArea)

	TextGrid textGrid() { return static_cast <TextGrid> (our data()); }

	integer d_draggedBoundary;

	void v_createMenuItems_help (EditorMenu menu)
		override;
	void v1_dataChanged (Editor sender) override {
		Melder_clipRight (& our textGridArea() -> selectedTier, our textGrid() -> tiers->size);   // crucial: before v_updateText (bug 2022-07-23)!
		our structFunctionEditor :: v1_dataChanged (sender);
		our textGridArea() -> functionChanged (our textGrid());
		if (our soundArea()) {
			our soundArea() -> functionChanged (nullptr);   // BUG: this function has not actually changed
			our soundAnalysisArea() -> functionChanged (our soundArea() -> function());
		}
	}
	void v_createMenus () override;
	bool v_mouseInWideDataView (GuiDrawingArea_MouseEvent event, double x_world, double globalY_fraction)
		override;
	double textGridAreaHeight () {
		if (! our soundArea())
			return 1.0;
		const integer numberOfTiers = our textGrid() -> tiers->size;
		const integer numberOfVisibleChannels =
				Melder_clippedRight (our soundArea() -> soundOrLongSound() -> ny, 8_integer);
		const bool showAnalysis = our soundAnalysisArea() -> hasContentToShow();
		const double automaticHeight = numberOfTiers / (2.0 * numberOfVisibleChannels +
				numberOfTiers * ( showAnalysis ? 1.8 : 1.3 ));
		const double textGridHeight = our instancePref_textGridHeight();
		return ( textGridHeight > 0.0 ? Melder_clipped (0.05, textGridHeight, 0.9) : automaticHeight );
	}
	double soundAreaHeight () {
		const double textGridHeight = our textGridAreaHeight();
		if (! our soundArea() || ! our soundAnalysisArea() -> hasContentToShow())
			return 1.0 - textGridHeight;
		const double soundHeight = our instancePref_soundHeight();
		double result = ( soundHeight > 0.0 ?
				Melder_clipped (0.1, soundHeight, 0.9) : 0.5 * (1.0 - textGridHeight) );
		if (result > 1.0 - textGridHeight - 0.05)
			result = 1.0 - textGridHeight - 0.05;
		return result;
	}
	void v_distributeAreas () override {
		if (our soundArea()) {
			const double textGridHeight = our textGridAreaHeight();
			our textGridArea() -> setGlobalYRange_fraction (0.0, textGridHeight);
			if (our soundAnalysisArea() -> hasContentToShow()) {
				const double soundBottom = 1.0 - our soundAreaHeight();
				our soundAnalysisArea() -> setGlobalYRange_fraction (textGridHeight, soundBottom);
				our soundArea() -> setGlobalYRange_fraction (soundBottom, 1.0);
			} else {
				our soundAnalysisArea() -> setGlobalYRange_fraction (textGridHeight, textGridHeight);
				our soundArea() -> setGlobalYRange_fraction (textGridHeight, 1.0);
			}
		} else {
			our textGridArea() -> setGlobalYRange_fraction (0.0, 1.0);
		}
	}
	void v_draw () override {
		if (our soundArea()) {
			FunctionArea_prepareCanvas (our soundArea().get());
			if (our soundAnalysisArea() -> instancePref_pulses_show())
				our soundAnalysisArea() -> v_draw_analysis_pulses ();
			FunctionArea_drawInside (our soundArea().get());
			if (our soundAnalysisArea() -> hasContentToShow()) {
				FunctionArea_prepareCanvas (our soundAnalysisArea().get());
				our soundAnalysisArea() -> v_draw_analysis ();
			}
		}
		FunctionArea_drawOne (our textGridArea().get());
	}
	bool v_hasText () override { return true; }
	bool v_hasSelectionViewer () override { return true; }
	void v_drawSelectionViewer () override {
		TextGridArea_drawSelectionViewer (our textGridArea().get());
	}
	void v_drawRealTimeSelectionViewer (double /* time */) override {
		TextGridArea_drawSelectionViewer (our textGridArea().get()); // avoid discontinuity during play
	}

	void v_clickSelectionViewer (double x_fraction, double y_fraction) override {
		TextGridArea_clickSelectionViewer (our textGridArea().get(), x_fraction, y_fraction);
	}
	conststring32 v_selectionViewerName ()
		override { return U"IPA chart"; }
	void v_play (double startTime, double endTime) override {
		if (our soundArea())
			SoundArea_play (our soundArea().get(), startTime, endTime);
	}
	void v_drawLegends () override {
		FunctionArea_drawLegend (our textGridArea().get(),
			FunctionArea_legend_TEXTGRID U" ##modifiable TextGrid",
			DataGui_defaultForegroundColour (our textGridArea().get(), false)
		);
		if (our soundArea()) {
			const bool pulsesAreVisible = our soundAnalysisArea() -> hasPulsesToShow ();
			FunctionArea_drawLegend (our soundArea().get(),
				FunctionArea_legend_WAVEFORM U" %%non-modifiable copy of sound",
				DataGui_defaultForegroundColour (our soundArea().get(), false),
				pulsesAreVisible ? FunctionArea_legend_POLES U" %%derived pulses" : nullptr,
				Melder_GREY
			);
			SoundAnalysisArea_drawDefaultLegends (our soundAnalysisArea().get());
		}
	}

	#include "TextGridEditor_prefs.h"
};

autoTextGridEditor TextGridEditor_create (conststring32 title, TextGrid grid,
	SampledXY sound,   // either a Sound or a LongSound, or null
	SpellingChecker spellingChecker,
	conststring32 callbackSocket
);

/* End of file TextGridEditor.h */
#endif
