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

package org.phalanxdev.mi.utils;

/**
 * Interface for logging implementations
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public interface IMILogAdapter {

  boolean isBasic();

  boolean isDebug();

  boolean isDetailed();

  boolean isError();

  boolean isRowLevel();

  void logBasic(String message);

  void logBasic(String message, Object... aux);

  void logDetailed(String message);

  void logDetailed(String message, Object... aux);

  void logMinimal(String message);

  void logMinimal(String message, Object... aux);

  void logDebug(String message);

  void logDebug(String message, Object... aux);

  void logError(String message);

  void logError(String message, Object... aux);
}
