package com.twitter.botmaker.function.collection;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode2;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {"MapRemove", "DictRemove"},
    argTypes = {
        "Map<A, B>",
        "Collection<A>"
    },
    arguments = {
        "dict",
        "keys"
    },
    returnType = "Map<A, B>",
    description = "Makes a new dict with given keys removed from the given dict \n",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "MapRemove(ToMap(123, \"asdf\", 456, \"jkl;\"), ToSet([123, 789]))"
    }
)
public class MapRemove extends FunctionNode2<Runtime, Map, Collection> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPB = Type.newGenericType();
  private static final Type TPM = Type.mapOf(TPA, TPB);
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(TPM, Type.collectionOf(TPA)),
      TPM);

  public MapRemove(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    super(exprText, children);
  }


  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  @SuppressWarnings("unchecked")
  protected Object apply(
      Context<Runtime> context,
      Map map,
      Collection collection
  ) {
    Set set = ImmutableSet.copyOf(collection);
    ImmutableMap.Builder builder = ImmutableMap.builder();
    for (Map.Entry entry : (Set<Map.Entry>) map.entrySet()) {
      if (!set.contains(entry.getKey())) {
        builder.put(entry);
      }
    }
    return builder.build();
  }
}
