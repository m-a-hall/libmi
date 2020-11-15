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

import org.phalanxdev.mi.PMIEngine;
import org.phalanxdev.mi.Scheme;
import org.phalanxdev.mi.SupervisedScheme;
import org.phalanxdev.mi.UnsupportedSchemeException;
import weka.core.WekaPackageClassLoaderManager;

import java.util.List;

/**
 * Engine for Spark MLlib. Uses the MLlib integration provided by Weka's distributedWekaSparkDev package.
 *
 * @author Mark Hall (mhall{[at]}waikato.ac.nz)
 * @version $Revision: $
 */
public class MLlibEngine extends PMIEngine {

  /** Name of the engine */
  public static final String ENGINE_NAME = "Spark - MLlib";

  /** Engine class */
  public static final String ENGINE_CLASS = MLlibEngine.class.getCanonicalName();

  /**
   * Get the name of the engine
   *
   * @return the name of the engine
   */
  @Override public String engineName() {
    return ENGINE_NAME;
  }

  /**
   * Returns true if the Spark MLlib engine is available
   *
   * @param messages a list to store error messages/info in
   * @return true if the Spark engine is available
   */
  @Override public boolean engineAvailable( List<String> messages ) {

    try {
      WekaPackageClassLoaderManager.forName( "weka.classifiers.mllib.MLlibClassifier" );
      return true;
    } catch ( ClassNotFoundException e ) {
      messages.add( e.getMessage() );
    }

    return false;
  }

  /**
   * Returns true if the named scheme is supported by the Spark MLlib engine
   *
   * @param schemeName the name of the scheme to check
   * @return true if the named scheme is supported
   */
  @Override public boolean supportsScheme( String schemeName ) {
    return SupervisedScheme.s_defaultClassifierSchemeList.contains( schemeName ) && !MLlibScheme.s_excludedSchemes.contains( schemeName );
  }

  /**
   * Get an instance of {@code Scheme} that encapsulates the named scheme in Spark MLlib
   *
   * @param schemeName the name of the scheme to get
   * @return a {@code Scheme} object
   * @throws UnsupportedSchemeException
   */
  @Override public Scheme getScheme( String schemeName ) throws UnsupportedSchemeException {
    try {
      return MLlibScheme.getSupervisedMLlibScheme( schemeName );
    } catch ( Exception ex ) {
      throw new UnsupportedSchemeException( ex );
    }
  }
}
