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

import java.util.ArrayList;
import java.util.List;
import weka.core.Environment;

/**
 * Default implementation of variables that uses Weka's Environment class
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public class DefaultVariables implements IMIVariableAdaptor {

  protected Environment m_vars = Environment.getSystemWide();

  public DefaultVariables(){}

  public DefaultVariables(Environment vars) {
    m_vars = vars;
  }

  @Override
  public List<String> listVariables() {
    return new ArrayList<>(m_vars.getVariableNames());
  }

  @Override
  public String getVariable(String varName) {
    return m_vars.getVariableValue(varName);
  }
}
