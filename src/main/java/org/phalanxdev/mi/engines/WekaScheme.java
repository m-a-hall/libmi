/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.phalanxdev.mi.engines;

import org.phalanxdev.mi.UnsupportedSchemeException;
import org.phalanxdev.mi.Scheme;
import org.phalanxdev.mi.SupervisedScheme;

import java.util.Arrays;
import java.util.List;

/**
 * Scheme implementation for WEKA.
 *
 * @author Mark Hall (mhall{[at]}waikato.ac.nz)
 * @version $Revision: $
 */
public abstract class WekaScheme {

  protected static List<String> s_excludedSchemes = Arrays.asList( "Deep learning network",
      "Extreme gradient boosting classifier", "Extreme gradient boosting regressor");

  /**
   * Static factory method for obtaining a {@code Scheme} instance that encapsulates a WEKA implementation of the
   * named scheme.
   *
   * @param schemeName the name of the scheme to get
   * @return a {@code Scheme} object
   * @throws UnsupportedSchemeException if a problem occurs
   */
  protected static Scheme getSupervisedWekaScheme( String schemeName ) throws UnsupportedSchemeException {
    if ( SupervisedScheme.s_defaultClassifierSchemeList.contains( schemeName ) && !s_excludedSchemes
        .contains( schemeName ) ) {
      return new WekaClassifierScheme( schemeName );
    } else {
      // TODO other types of schemes - clusterers etc.
    }

    throw new UnsupportedSchemeException( "Weka engine does not support scheme: " + schemeName );
  }
}
