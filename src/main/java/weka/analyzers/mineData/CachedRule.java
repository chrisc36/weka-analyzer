package weka.analyzers.mineData;

import java.util.BitSet;

/**
 * Interface for 'rules' or expressions that evaluate to true or false for each
 * instance in an indexed dataset and have cached the results in a BitSet.
 */
public interface CachedRule {

    /**
     * Gets a BitSet such the ith bit set iff this Rule would return
     * true for the ith instance in the dataset this was built from.
     *
     * @return BitSet with set bits corresponding to instances this Rule covers
     */
    public BitSet covered();
}
