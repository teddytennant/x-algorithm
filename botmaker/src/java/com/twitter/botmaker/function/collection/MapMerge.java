package com.twitter.botmaker.function.collection;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {"MapMerge", "DictMerge"},
    argTypes = {"[Map<A, B>...]"},
    arguments = {"dict[i]"},
    returnType = "Map<A, B>",
    description = "Merges two dicts into a new one",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "MapMerge(\n"
            + "ToMap(123, \"asdf\", 456, \"jkl;\"),\n"
            + "ToMap(456, \"xxxx\", 789, \"yyyy\"))"
    }
)
public class MapMerge extends FunctionNode<Runtime> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPB = Type.newGenericType();
  private static final Type MAP = Type.mapOf(TPA, TPB);
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(),
      MAP,
      MAP);

  public MapMerge(String exprText, ImmutableList<ASTNode> children)
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
  protected Object apply(Context<Runtime> context, List<Object> args) {
    HashMap result = Maps.newHashMap();
    for (Object arg : args) {
      for (Map.Entry entry : (Set<Map.Entry>) ((Map) arg).entrySet()) {
        result.put(entry.getKey(), entry.getValue());
      }
    }
    return ImmutableMap.copyOf(result);
  }
}
