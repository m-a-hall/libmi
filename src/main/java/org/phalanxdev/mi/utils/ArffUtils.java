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

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;

/**
 * Utilities for creating ARFF data.
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public class ArffUtils {

  /**
   * Create a header (only attribute metadata) Instances object
   *
   * @param relationName the relation name for the data
   * @param attNames attribute names as a list of strings
   * @param attTypes a list of types for the attributes - elements can be a Number for a numeric
   * attribute, an empty list for a string attribute or a list of strings for a nominal attribute
   * @param className optional name of the attribute/column to set as the class attribute in the *
   * resulting Instances object
   * @return a header set of instances
   */
  @SuppressWarnings("unchecked")
  public static Instances simpleSpecHeader(String relationName, List<String> attNames,
      List<Object> attTypes, String className) throws Exception {
    if (attNames.size() != attTypes.size()) {
      throw new Exception(
          "Number of attribute names does not match the number of specified types!");
    }
    ArrayList<Attribute> atts = new ArrayList<>();
    for (int i = 0; i < attNames.size(); i++) {
      if (attTypes.get(i) instanceof Number) {
        atts.add(new Attribute(attNames.get(i)));
      } else if (attTypes.get(i) instanceof List) {
        if (((List<String>) attTypes.get(i)).size() == 0) {
          atts.add(new Attribute(attNames.get(i), (List<String>) null));
        } else {
          atts.add(new Attribute(attNames.get(i), (List<String>) attTypes.get(i)));
        }
      }
    }

    Instances result = new Instances(relationName, atts, 0);
    if (className != null && className.length() > 0 && result.attribute(className) != null) {
      if (result.attribute(className).isString()) {
        throw new Exception("Class attribute can only be numeric or nominal");
      }
      result.setClass(result.attribute(className));
    }
    return result;
  }

  /**
   * Convert a string containing csv data to an Instances object
   *
   * @param csvInput the csv data as a string
   * @param csvLoaderOpts optional command line options to the CSVLoader
   * @param className optional name of the attribute/column to set as the class attribute in the
   * resulting Instances object. If null, an empty string, or does not exist as a column in the csv
   * data then no class is set
   * @return an Instances object
   * @throws Exception if a problem occurs
   */
  public static Instances csvDataToInstances(String csvInput, String csvLoaderOpts,
      String className) throws Exception {
    CSVLoader loader = new CSVLoader();
    if (csvLoaderOpts != null && csvLoaderOpts.length() > 0) {
      loader.setOptions(Utils.splitOptions(csvLoaderOpts));
    }

    loader.setSource(new ByteArrayInputStream(csvInput.getBytes("UTF-8")));
    Instances result = loader.getDataSet();
    if (className != null && className.length() > 0 && result.attribute(className) != null) {
      if (result.attribute(className).isString()) {
        throw new Exception("Class attribute can only be numeric or nominal");
      }
      result.setClass(result.attribute(className));
    }

    return result;
  }
}
